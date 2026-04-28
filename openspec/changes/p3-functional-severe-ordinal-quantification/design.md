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
best supportable product
  |
  +-- first: high-sensitivity current-data severe triage
  +-- second: deployable ordinal burden bands if validated
  +-- third: six-bin ordinal comparator if stable and useful
  +-- never: unsupported scalar burden or external validation claim
```

## Explicit Decisions

- Primary implementation module: `src/eq/quantification/functional_p3.py`.
- Primary runtime root: `burden_model/functional_quantification_p3/`.
- Pipeline integration point: after P2 severe-aware outputs are available in `src/eq/quantification/pipeline.py`.
- No internal locked test split is required; scarce current labeled data are preserved for grouped development validation and final refit.
- Primary product objective: reduce adjudicated `score >= 2` severe false negatives under a current-data/source-sensitive claim boundary.
- Available quantification labels are image-level aggregate labels. Multi-glomerulus images must not be treated as if every component has the same per-glomerulus truth.
- Secondary product objective: test ordinal burden bands before declaring ordinal burden unsupported.
- Severe-risk gates outrank scalar or exact-score burden metrics.
- Three-band ordinal is the first ordinal product candidate; six-bin exact ordinal is diagnostic unless it clears grouped out-of-fold gates.
- Morphology features may be predictive covariates even when blocked for biological closed-lumen claims.
- Learned ROI and embedding-heavy neural features must be tested as full P3 candidate lanes, not only mentioned as theoretical comparators.
- Learned/embedding-heavy candidates can win only if they improve grouped development out-of-fold severe or ordinal-band gates without leakage, nonfinite outputs, or unacceptable source/cohort sensitivity.
- P3 may run autonomous threshold/model/feature loops within the bounded task list and stop rules.
- P3 must not require new manual review during implementation.
- P3 must not start segmentation-backbone work unless the final evidence localizes the bottleneck to mask geometry.
- P3 durable logging remains caller-owned through the existing `endotheliosis_quantification` run-config surface.

## Product Ladder

P3 should climb this ladder and stop at the highest rung supported by evidence:

| Rung | Product | Claim Boundary | Promotion Gate |
| --- | --- | --- | --- |
| 1 | Current-data severe-risk triage | Flags likely grade 2/3 images for review | high severe recall with finite grouped out-of-fold evidence |
| 2 | Current-data ordinal burden bands | Predicts none/low, mild/mid, severe bands | severe-band recall plus band balanced accuracy |
| 3 | Six-bin ordinal comparator | Diagnostic six-score rubric probabilities/sets | beats baselines with calibrated adjacent errors |
| 4 | Scalar burden | Continuous grade-equivalent score | only if ordinal/scalar gates pass; unlikely from P2 evidence |
| 5 | README-facing MR TIFF deployment | Runs supported segmentation-to-quantification test on the `vegfri_mr` whole-field TIFF cohort | quantification gate plus MR deployment test pass |
| 6 | External validation | Generalizes to new cohort/source | out of scope; current severe positives are source-confounded |

The implementation should not stop at rung 1 if rung 2 is feasible. It should attempt rung 2 and rung 3, but severe false-negative reduction is the main product priority.

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

P3 root: `burden_model/functional_quantification_p3/`.

Required artifacts:

- `INDEX.md`
- `summary/executive_summary.md`
- `summary/final_product_verdict.json`
- `summary/final_product_verdict.md`
- `summary/model_selection_table.csv`
- `summary/development_oof_metrics.csv`
- `summary/ordinal_feasibility.json`
- `summary/severe_threshold_selection.json`
- `summary/artifact_manifest.json`
- `splits/development_folds.csv`
- `feature_sets/p3_feature_matrix.csv`
- `feature_sets/p3_feature_diagnostics.json`
- `predictions/development_oof_predictions.csv`
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

## MR TIFF Deployment Scope

P3 is primarily a quantification-model change. It becomes an end-to-end deployable product only if it also proves that the selected quantification model can run downstream of the supported segmentation pipeline on the `vegfri_mr` whole-field TIFF cohort available under the runtime manifest.

The MR deployment test must use a supported current-namespace segmentation artifact. It must process whole-field TIFFs by tiling, segmenting glomeruli, merging tile predictions into the whole-field coordinate frame, filtering connected components by area and quality, extracting accepted ROI image/mask records, computing the exact P3 inference feature schema, loading the final quantification model, writing per-ROI predictions, aggregating to image-level median/summary outputs, and producing a human-readable report. When workbook human image-level medians are available, it must report human-vs-inferred concordance; when labels are absent, it is only a technical deployment smoke, not an accuracy test. Passing a model on precomputed feature tables alone is not enough for README-facing deployment.

If a supported segmentation artifact or MR TIFF input is unavailable, P3 must say that explicitly. In that case, a model may still be `deployable_current_data_severe_triage` or `deployable_current_data_ordinal_bands` as a quantification artifact, but README-facing end-to-end deployment is blocked until the MR deployment test exists.

## Final Verdict States

`deployable_current_data_severe_triage`

- severe recall gate passes
- ordinal gates do not pass or are secondary
- output is a high-sensitivity review triage label

`deployable_current_data_ordinal_bands`

- severe recall gate passes
- three-band or four-band ordinal gate passes
- output is ordered band prediction plus severe-risk flag

`readme_facing_deployable_current_data_model`

- severe or ordinal quantification gate passes
- final model artifact and inference schema are written
- MR TIFF segmentation-to-quantification deployment test passes
- README-facing report states current-data/source-sensitive claim boundary

`model_ready_pending_mr_tiff_deployment_smoke`

- severe or ordinal quantification gate passes
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
