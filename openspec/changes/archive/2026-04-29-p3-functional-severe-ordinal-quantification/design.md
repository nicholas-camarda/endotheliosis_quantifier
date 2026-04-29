## Context

P2 archived the first severe-aware estimator and synced the severe-aware specification. Its final full-cohort result was limited: original `score >= 2` recall was 0.146 and the adjudicated-label rerun recall was 0.113. However, the post-P2 read-only screen showed that the current data are not signal-free. Using existing morphology features with a class-balanced severe objective produced subject-heldout AUROC `0.829` and average precision `0.326`. At fixed or diagnostic operating thresholds, the same screen reached recall `0.676` with precision `0.262` at threshold `0.5`, recall `0.803` with precision `0.288` near threshold `0.477`, and recall `0.944` with precision `0.211` at threshold `0.3`.

This does not prove that morphology-only logistic regression is the final product. It proves a more important planning point: the current data are not signal-free, and P2 partly failed because it optimized/report-selected a conservative severe classifier rather than a high-sensitivity grade 2/3 triage objective. P3 must therefore make high-sensitivity severe detection an explicit operational target and then control false positives with review artifacts, threshold reporting, and reliability labels.

This means the next product should not be framed as "make the old P2 model pass." It should be framed as:

```text
available evidence
  |
  +-- masks appear adequate for glomerulus vs non-glomerulus
  +-- severe labels are noisy but partially adjudicated
  +-- score >= 2 positives are source-sensitive and all in vegfri_dox
  +-- deterministic morphology is not validated biological proof
  +-- morphology still carries predictive severe signal
  +-- learned/embedding-heavy features are risky but not disproven for P3
  +-- labels are image-level aggregates, not per-glomerulus truth
  |
  v
main product target
  |
  +-- deployable MR TIFF segmentation-to-quantification grade model
      |
      +-- candidate output: ordinal grade bands if validated
      +-- candidate output: calibrated score-like or scalar grade estimate only if gates pass
      +-- fallback output: high-sensitivity severe-risk triage if richer grade outputs fail
      +-- diagnostic only: six-bin comparator or scalar output unless gates pass
      +-- never: unsupported scalar burden or external validation claim
```

## Explicit Decisions

- Primary implementation module: `src/eq/quantification/endotheliosis_grade_model.py`.
- Primary runtime root: `burden_model/endotheliosis_grade_model/`.
- Pipeline integration point: after P2 severe-aware outputs are available in `src/eq/quantification/pipeline.py`.
- No internal locked test split is required; scarce current labeled data are preserved for grouped development validation and final refit.
- Primary product objective: deploy a defensible endotheliosis grade quantification model on the MR TIFF workflow under a current-data/source-sensitive claim boundary.
- Available quantification labels are image-level aggregate labels. Multi-glomerulus images must not be treated as if every component has the same per-glomerulus truth.
- Row-level optional feature tables, including learned ROI and morphology feature tables, join to P3 candidate rows by `subject_image_id` and `glomerulus_id`. Missing or duplicate row-level join keys are hard blockers, not image-level fallback joins.
- Primary safety objective: preserve useful grade 2/3 severe sensitivity so the deployed grade model does not look acceptable while missing clinically/biologically important severe disease.
- Secondary product objective: test ordinal burden bands before falling back to severe-risk triage or diagnostic-only outputs.
- Severe-risk gates are safety gates for deployability; they do not replace the MR TIFF deployment and grade-quantification objective.
- Three-band ordinal is the first ordinal product candidate; six-bin exact ordinal is diagnostic unless it clears grouped out-of-fold gates.
- Morphology features may be predictive covariates even when blocked for biological closed-lumen claims.
- Learned ROI and embedding-heavy neural features must be tested as full P3 candidate lanes, not only mentioned as theoretical comparators.
- Learned/embedding-heavy candidates can win only if they improve grouped development out-of-fold severe or ordinal-band gates without leakage, nonfinite outputs, or unacceptable source/cohort sensitivity.
- P3 may run autonomous threshold/model/feature loops within the bounded task list and stop rules.
- P3 must not require new manual review during implementation.
- P3 must not start segmentation-backbone work unless the final evidence localizes the bottleneck to mask geometry.
- P3 durable logging remains caller-owned through the existing `endotheliosis_quantification` run-config surface.
- Shared model-selection mechanics are governed in `src/eq/quantification/modeling_contracts.py`. P3 must use those helpers for finite feature matrices, recall-targeted threshold selection, warning capture, JSON/manifest writing, and grouped-development metric labels instead of recreating those contracts in the selector.
- Candidate family identity must be source-truthful. A candidate named `embedding_*` is invalid unless actual embedding columns are present, and a candidate named `learned_*` is invalid unless learned-feature columns are present.
- MR TIFF promotion requires an inference schema whose feature columns can be computed from the segmentation-to-ROI deployment path. Learned/embedding-heavy feature schemas remain diagnostic unless the deployment path computes those exact columns.
- A Dox scored-no-mask smoke set is the pre-MR deployment bridge. It is built from Dox `scored_only` manifest rows that resolve exactly once to Label Studio upload images, have nonmissing scores, and have no duplicate/conflicting source-image evidence. Clean images are copied to the runtime cohort tree, stored in canonical `image_path`, and permitted for this workflow by Dox smoke eligibility/status columns.

## Product Ladder

The grade-model workflow should climb this ladder and stop at the highest rung supported by evidence:

| Rung | Product | Claim Boundary | Promotion Gate |
| --- | --- | --- | --- |
| 1 | README-facing MR TIFF grade model | Runs segmentation-to-quantification on `vegfri_mr` and reports the best supported grade output | selected grade-output gate plus severe safety gate plus MR deployment test pass |
| 2 | README-facing MR TIFF severe-risk triage | Runs segmentation-to-quantification on `vegfri_mr` and flags likely grade 2/3 for review | severe safety gate plus MR deployment test pass when ordinal bands fail |
| 3 | Model-ready grade artifact pending MR deployment | Current-data grade or severe gate passes, but MR deployment is blocked or incomplete | model artifacts and inference schema pass, MR deployment not yet passable |
| 4 | Diagnostic current-data grade model | Useful development evidence but not deployable | reports failed gate and best failed candidate |
| 5 | Current-data insufficient | No candidate beats baseline gates after bounded attempts | writes additional-data recommendation |
| 6 | External validation | Generalizes to new cohort/source | out of scope; current severe positives are source-confounded |

The preferred product is an MR TIFF deployable grade model. The selected grade output may be ordinal bands, a calibrated score-like output, or another gated representation if it is the best defensible result from the candidate loop. Severe-risk triage is the fallback product if richer grade outputs fail but severe-sensitive deployment is still useful and honest. Severe false-negative reduction remains a safety gate and fallback value proposition, not the main product priority.

## Development And Final-Fit Strategy

P3 should not hold out scarce current labeled scored/mask-paired subjects as a fixed internal test set. It should use grouped development validation for model selection and then refit the selected model on all eligible labeled rows:

```text
all current labeled scored/mask-paired subjects
  |
  +-- grouped development folds
  |     subject-heldout folds for OOF metrics, threshold selection, calibration, feature selection
  |
  +-- final refit
        selected feature/model/threshold recipe fit on all eligible labeled rows
        then applied to MR TIFF deployment data
```

Rules:

- Development folds must be subject-heldout.
- Fold assignment must be deterministic and written before candidate fitting.
- Fold assignment should stratify by subject-level score profile: max score, presence of score 2/3, and score-bin coverage where feasible.
- All threshold selection, feature selection, dimensionality reduction, calibration, and model selection must be performed inside development folds or inner development splits.
- The final model may be refit on all eligible labeled rows only after the selected recipe is fixed.
- P3 must label current-data metrics as grouped out-of-fold development estimates, not independent heldout estimates.

## Autonomous Execution Loop

P3 apply should run without user intervention by using explicit branch rules:

```text
1. reconstruct inputs
2. assign grouped development folds
3. build feature matrix
4. run baseline candidates
5. run severe-focused candidates
6. run ordinal/banded candidates
7. inspect failures automatically
8. add bounded feature transforms if justified
9. rerun candidates
10. select best supportable product
11. write final verdict and review artifacts
```

Allowed autonomous choices:

- If P2 adjudications exist, use them as the severe target. If not, use original `score >= 2` and label the verdict unadjudicated.
- If `inputs.label_overrides` is configured, apply explicit reviewed rubric scores before ROI extraction, embedding extraction, burden modeling, learned ROI modeling, source-aware modeling, severe-aware modeling, and P3 selection. The override file must be row-level and keyed by `subject_image_id`; supported score columns are `rubric_score`, `reviewer_score`, `reviewer_grade`, or `score`.
- Reviewed rubric overrides are a first-class label source, not projected labels. The pipeline must write `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json` with original scores, override scores, changed-row counts, severe-boundary changes, reviewer metadata when present, and the claim boundary `explicit reviewer label overrides only; no inferred labels applied`.
- Missing override files, duplicate `subject_image_id` rows, unrecognized scores, nonnumeric scores, or references to absent scored examples are hard failures. The workflow must not silently ignore or partially apply a requested override file.
- If six-bin ordinal support is too weak, collapse to three-band and four-band outputs before declaring ordinal impossible.
- If the severe model achieves high recall only at low precision, report it as a triage product, not a definitive classifier.
- If learned ROI or embedding features overfit development folds or fail source-sensitivity gates, exclude them from final selection but retain them in the ablation report.
- If morphology features help prediction but fail biological readiness, use them as predictive covariates only and explicitly block mechanistic closed-lumen claims.
- If all gates fail, produce an insufficiency verdict rather than asking the user for manual rescue.

## Candidate Families

Baseline families:

- majority/empirical prior baselines
- P2 selected severe-aware candidate as baseline comparator
- ROI/QC logistic severe gate
- morphology logistic severe gate
- ROI/QC + morphology logistic severe gate
- three-band ordinal logistic or calibrated multiclass logistic
- six-bin ordinal comparator

Severe-focused families:

- class-balanced logistic regression
- calibrated logistic regression with regularization sweep
- linear SVM or SGD logistic when numerically stable
- random forest / extra trees with class weighting as exploratory
- histogram gradient boosting with shallow constraints as exploratory
- threshold-selected high-recall variants using inner development folds
- bounded regularization sweeps for logistic severe gates
- source-truthful candidate registration so missing source-specific columns cannot silently downgrade a learned or embedding candidate into a morphology-only candidate

Feature families:

- existing ROI/QC
- existing deterministic morphology
- existing learned ROI simple QC and encoder features as required gated candidates
- reduced embeddings as required gated candidates after dimensionality control
- embedding-heavy neural candidates using PCA/feature selection/regularized models so high-dimensional embeddings are tested without uncontrolled overfit
- hybrid learned-plus-morphology and learned-plus-ROI candidates, gated by grouped out-of-fold performance and source-sensitivity diagnostics
- derived severe-focused transforms:
  - robust z-scored morphology
  - interactions among open-lumen, collapsed/slit-like, RBC, and quality/confounder signals
  - per-subject aggregate context features where used only for subject-level outputs
  - quality-gated variants that downweight or label unreliable rows
  - multi-component aggregate features for image-level labels: component count, component area summaries, feature mean/median/max/upper quantile/spread, and high-risk component summaries

Row-level feature tables:

- `learned_roi/feature_sets/learned_roi_features.csv` must preserve `glomerulus_id` from the input embedding/ROI rows when available.
- `primary_burden_index/feature_sets/morphology_features.csv` and `learned_roi/feature_sets/learned_roi_features.csv` are ROI-level sources for P3.
- P3 must join those ROI-level sources on `subject_image_id` and `glomerulus_id`; a table that only has `subject_image_id` is not unique enough for multi-glomerulus images.
- Duplicate optional-table keys are unusable and must be recorded as hard blockers rather than allowed to crash later or multiply rows.

## Multi-Glomerulus Aggregate Labels

Some scored inputs contain multiple glomeruli while the available score is an image-level average or summary. That can hurt the model because a single label is being attached to a union ROI that may contain heterogeneous components. A severe glomerulus mixed with milder glomeruli can be averaged down, and a mild image with one suspicious component can look severe if the model treats every component as carrying the image label.

P3 should handle this without new manual work by treating the score as an image-level aggregate target. It should preserve the current union-ROI path, but also test aggregate-aware features derived from existing masks:

- number of connected glomerulus components
- component area distribution
- component-level morphology/embedding summaries
- mean, median, maximum, upper quantile, and spread of severe-risk component features
- image-level predictions produced by aggregating component-level risk scores

The workflow must not create fake per-glomerulus labels from an image-average score. Component-level predictions are allowed as intermediate or diagnostic outputs only when the selected product is evaluated against the image-level aggregate target.

If aggregate-aware candidates do not improve grouped out-of-fold behavior, P3 should record this as a limitation and continue with the best image-level product rather than requiring recropping or new labels.

## Learned And Embedding-Heavy Candidate Plan

P3 should not throw away neural-network-derived features. It should test them under rules strict enough to prevent self-deception:

```text
learned/embedding feature source
  |
  +-- existing learned_roi feature table
  +-- current glomeruli encoder embeddings
  +-- reduced embedding subsets
  |
  v
dimensionality control
  |
  +-- PCA or supervised feature selection inside development folds only
  +-- regularized logistic / calibrated linear models
  +-- tree comparators only as exploratory
  |
  v
gates
  |
  +-- grouped development performance
  +-- grouped out-of-fold severe recall / ordinal bands
  +-- finite output status
  +-- source/cohort predictability and residual checks
  |
  v
select if real, reject if source/style/overfit
```

Required learned/embedding-heavy candidate tracks:

- `learned_roi_severe_gate`: learned ROI features only, regularized high-sensitivity severe gate.
- `embedding_reduced_severe_gate`: reduced encoder embeddings only, regularized high-sensitivity severe gate.
- `learned_morphology_severe_gate`: learned ROI plus morphology features.
- `embedding_morphology_severe_gate`: reduced embeddings plus morphology features.
- `learned_three_band_ordinal`: learned ROI features for `none_low`, `mild_mod`, `severe`.
- `embedding_three_band_ordinal`: reduced embeddings for the three-band target.
- `hybrid_three_band_ordinal`: learned or reduced embeddings plus morphology/ROI features.

These candidates are allowed to win. The reason they are gated is not that neural features are invalid; it is that existing artifacts showed prior learned/embedding-heavy ordinal candidates had strong apparent fit and poor subject-heldout behavior, plus cohort predictability. P3 must retest them on the actual P3 target with grouped out-of-fold evidence and MR deployment behavior.

## Ordinal Feasibility Position

We are not positive that ordinal burden is impossible. The evidence says precise scalar burden is not currently justified, and exact six-bin prediction is likely difficult. P3 should therefore test ordinal outputs in this order:

1. three-band: none/low vs mild/mid vs severe
2. four-band: 0 vs 0.5 vs 1/1.5 vs 2/3
3. six-bin exact or prediction-set output

Reportability depends on grouped out-of-fold behavior and MR deployment behavior:

- If three-band passes, ship ordinal burden bands.
- If only severe band passes, ship severe triage only.
- If six-bin exact fails but prediction sets cover adjacent labels with useful width, keep six-bin as diagnostic evidence.
- If all ordinal forms fail, write explicit evidence that current data support severe triage only.

## Output Contract

## Output Architecture Amendment

Existing model families keep their existing output directories when they are rerun or consumed:

- `burden_model/primary_burden_index/`
- `burden_model/learned_roi/`
- `burden_model/source_aware_estimator/`
- `burden_model/severe_aware_ordinal_estimator/`

Newly fit model families must be first-class burden-model subtrees, not hidden under the final selector. Expected new model-family subtrees:

- `burden_model/three_band_ordinal_model/`
- `burden_model/four_band_ordinal_model/` when score support allows
- `burden_model/severe_triage_model/`
- `burden_model/aggregate_grade_model/`
- `burden_model/embedding_grade_model/`

Each first-class model-family subtree must use this navigable shape:

- `INDEX.md`
- `summary/`
- `diagnostics/`
- `predictions/`
- `model/`
- `evidence/`
- `internal/`

The final selector and deployment layer is `burden_model/endotheliosis_grade_model/`. It compares the model-family subtrees, records candidate coverage, writes the selected deployed model, and proves or blocks MR TIFF deployment. It must link back to each source model-family subtree rather than hiding family evidence only in `internal/`.

## Model-Family Diagnostics Contract

Every first-class model-family subtree must write diagnostics sufficient to explain whether the family was valid, failed, or selected:

- `diagnostics/input_support.json`
- `diagnostics/feature_diagnostics.json`
- `diagnostics/fold_diagnostics.json`
- `diagnostics/source_sensitivity.json`
- `diagnostics/gate_diagnostics.json`

Additional diagnostics are required when applicable:

- `diagnostics/embedding_source_predictability.json`
- `diagnostics/aggregate_label_diagnostics.json`
- `diagnostics/calibration_diagnostics.json`
- `diagnostics/threshold_selection_diagnostics.json`
- `diagnostics/mr_tiff_deployment_diagnostics.json`
- `diagnostics/hard_blockers.json`

Missing expected upstream artifacts for required families are hard failures, not ordinary skipped candidates.

## Final Selector Output Contract

Selector root: `burden_model/endotheliosis_grade_model/`.

Required artifacts:

- `INDEX.md`
- `summary/executive_summary.md`
- `summary/candidate_coverage_matrix.csv`
- `summary/final_product_verdict.json`
- `summary/final_product_verdict.md`
- `summary/model_selection_table.csv`
- `summary/development_oof_metrics.csv`
- `summary/ordinal_feasibility.json`
- `summary/severe_threshold_selection.json`
- `summary/input_artifact_index.json`
- `summary/artifact_manifest.json`
- `diagnostics/selector_diagnostics.json`
- `diagnostics/candidate_family_gate_diagnostics.json`
- `splits/development_folds.csv`
- `predictions/development_oof_predictions.csv`
- `predictions/final_model_training_predictions.csv`
- `internal/candidate_metrics.csv`
- `internal/candidate_configs.json`
- `internal/autonomous_loop_log.json`
- `evidence/error_review.html`
- `evidence/severe_false_negative_review.html`
- `evidence/ordinal_confusion_review.html`

Optional deployable model artifacts if a gate passes:

- `model/final_model.joblib`
- `model/final_model_metadata.json`
- `model/inference_schema.json`
- `model/deployment_smoke_predictions.csv`

Required deployment-smoke artifacts before README-facing promotion:

- `deployment/mr_tiff_smoke_manifest.csv`
- `deployment/mr_tiff_smoke_predictions.csv`
- `deployment/mr_tiff_smoke_report.html`
- `deployment/segmentation_quantification_contract.json`

The selected deployed model must be traceable from `summary/final_product_verdict.json`, `summary/model_selection_table.csv`, and `model/final_model_metadata.json` back to its source model-family subtree and diagnostics.

If final gates fail, stale deployable artifacts from prior runs must be removed from `burden_model/endotheliosis_grade_model/model/` so a diagnostic-only or insufficient verdict cannot be mistaken for a runnable final product.

## Repo-Wide Audit And Centralization Contract

Research-partner audit during apply found repeated local implementations of model-selection mechanics across quantification evaluators: grouped out-of-fold labels, threshold selection, warning handling, finite numeric preprocessing, and artifact manifests. P3 must not add another independent copy of those mechanics. The shared contract is:

- `GROUPED_DEVELOPMENT_METRIC_LABEL` is the canonical label for current-data grouped out-of-fold metrics.
- `to_finite_numeric_matrix()` owns coercion from DataFrame feature columns to finite model matrices.
- `choose_recall_threshold()` owns recall-targeted severe operating-threshold selection.
- `capture_fit_warnings()` owns warning capture for candidate fitting and final refits.
- `save_json()` and `build_artifact_manifest()` own common JSON and manifest writing.

This does not require broad refactoring of older P0/P1/P2 evaluators during P3, but P3 output and new tests must use the shared helpers, and the audit result must be recorded so future cleanup has a concrete target.

The selector reads existing artifact inputs from their existing locations:

- `burden_model/primary_burden_index/feature_sets/`
- `burden_model/primary_burden_index/candidates/`
- `burden_model/learned_roi/feature_sets/`
- `burden_model/learned_roi/candidates/`
- `burden_model/source_aware_estimator/summary/`
- `burden_model/severe_aware_ordinal_estimator/summary/`
- `burden_model/severe_aware_ordinal_estimator/evidence/`

The selector may write compact derived tables under `internal/` when needed for reproducibility, but no required model family may be represented only by selector-internal logs. Every newly fit required family must have a source subtree, diagnostics, metrics, predictions, and evidence paths listed in `summary/candidate_coverage_matrix.csv`.

`summary/candidate_coverage_matrix.csv` must include, at minimum:

- `family_id`
- `subtree_path`
- `required_or_exploratory`
- `run_status`
- `candidate_ids`
- `metrics_path`
- `diagnostics_path`
- `predictions_path`
- `gate_status`
- `selected`
- `failure_or_exclusion_reason`

## MR TIFF Deployment Scope

P3 is primarily a quantification-model change. It becomes an end-to-end deployable product only if it also proves that the selected quantification model can run downstream of the supported segmentation pipeline on the `vegfri_mr` whole-field TIFF cohort available under the runtime manifest.

Before MR TIFF deployment, P3 must run the smaller Dox scored-no-mask smoke stage when the master manifest contains rows with `eligible_dox_scored_no_mask_smoke = true`. This stage uses runtime-local images under `raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images/`, runs supported glomerulus segmentation, extracts predicted ROI records, computes the selected P3 inference schema, loads `model/final_model.joblib`, writes ROI-level predictions, aggregates to image-level summaries, and compares inferred image-level severe/grade behavior against the human Dox score. It is a pre-MR deployment bridge, not external validation and not a replacement for the MR TIFF deployment test.

The Dox smoke input is governed by `raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.csv`. The audit must resolve each Dox `scored_only` manifest row to exact Label Studio upload images, copy clean images into the runtime cohort tree, and flag duplicate source image names, conflicting source-image scores, missing scores, and unresolved/multi-match images. Only rows with exactly one resolved image, no duplicate source image name, no conflicting score, and a nonmissing score can enter `dox_scored_no_mask_smoke_manifest.csv`. The master manifest uses canonical `image_path` for the localized smoke image, keeps `mask_path` empty, and uses `eligible_dox_scored_no_mask_smoke` to permit this row only for the Dox smoke workflow.

If Dox smoke fails because severe recall is preserved but severe precision is unacceptable, P3 writes `dox_scored_no_mask_overcall_triage_queue.csv` and `dox_scored_no_mask_overcall_triage_report.html`. The queue is selected from deployment-computable Dox prediction fields, includes clustered false-positive representatives, highest-confidence false positives, threshold-boundary false positives, human severe references, and segmentation misses, and is intended to drive a bounded review of model overcalling before any MR TIFF deployment attempt.

When a reviewer completes the cluster-representative false-positive rows, P3 summarizes that review in `dox_scored_no_mask_overcall_review_diagnostic.json` and `dox_scored_no_mask_first12_review_interpretation.csv`. If at least eight cluster representatives are reviewed, at least five clearly usable ROIs are reviewed, and the clearly usable reviewed ROIs are predominantly non-severe with no clearly usable severe examples, the selected severe-risk candidate is rejected as a Dox overcaller. In that case P3 downgrades the verdict to `diagnostic_only_current_data_model`, records hard blockers, removes `model/final_model.joblib`, `model/final_model_metadata.json`, and `model/inference_schema.json`, and does not proceed to MR TIFF deployment.

The MR deployment test must use a supported current-namespace segmentation artifact. It must process whole-field TIFFs by tiling, segmenting glomeruli, merging tile predictions into the whole-field coordinate frame, filtering connected components by area and quality, extracting accepted ROI image/mask records, computing the exact P3 inference feature schema, loading the final quantification model, writing per-ROI predictions, aggregating to image-level median/summary outputs, and producing a human-readable report. When workbook human image-level medians are available, it must report human-vs-inferred concordance; when labels are absent, it is only a technical deployment smoke, not an accuracy test. Passing a model on precomputed feature tables alone is not enough for README-facing deployment.

If a supported segmentation artifact or MR TIFF input is unavailable, P3 must say that explicitly. In that case, a model may still be `model_ready_pending_mr_tiff_deployment_smoke` as a quantification artifact, but README-facing end-to-end deployment is blocked until the MR deployment test exists.

## Final Verdict States

`readme_facing_deployable_mr_tiff_grade_model`

- at least one non-triage grade-output quantification gate passes
- severe safety gate passes
- final model artifact and inference schema are written
- MR TIFF segmentation-to-quantification deployment test passes
- README-facing report states current-data/source-sensitive claim boundary

`readme_facing_deployable_mr_tiff_severe_triage`

- no richer non-triage grade output passes
- severe safety gate passes
- final model artifact and inference schema are written
- MR TIFF segmentation-to-quantification deployment test passes
- README-facing report states high-sensitivity review-triage claim boundary

`model_ready_pending_mr_tiff_deployment_smoke`

- ordinal grade-band or severe-triage gate passes
- final model artifact and inference schema are written
- MR TIFF deployment test cannot be completed because a supported segmentation artifact or MR TIFF input is missing or fails
- README-facing deployment language remains blocked

`diagnostic_only_current_data_model`

- development evidence is promising but MR deployment evidence fails or is too incomplete
- no deployable claim

`current_data_insufficient`

- no candidate beats baselines under grouped development evidence
- output includes required additional-data estimate and failure localization

## Validation Plan

Focused validation:

- unit tests for grouped fold determinism and score-bin reporting
- unit tests for no subject leakage across development folds
- unit tests for threshold selection using only development folds
- unit tests for severe gate promotion logic
- unit tests for ordinal band promotion logic
- unit tests for insufficiency verdict
- manifest completeness test
- executive summary completeness test
- focused quantification pipeline integration test

Runtime validation:

- run P3 evaluator on the current full-cohort transfer P0 adjudicated output
- inspect final verdict, grouped out-of-fold metrics, severe false negatives, ordinal confusion, MR deployment artifacts, and manifest
- if quantification gates pass, run the MR TIFF segmentation-to-quantification deployment test before any README-facing claim
- run OpenSpec strict validation and explicitness check

## Residual Risks

- Severe positives are source-confounded: all adjudicated severe rows are currently from `vegfri_dox`.
- Grouped out-of-fold development estimates may be high variance because there are only 71 adjudicated severe-positive rows.
- Image-level average labels on multi-glomerulus inputs limit per-glomerulus interpretation unless future per-glomerulus labels are created.
- Morphology features can be predictive while still biologically non-validated.
- False positives may be numerous if the product prioritizes high sensitivity.
- A deployable current-data product is possible; an externally validated burden estimator is not supported by current evidence.
