## Why

P2 proved that the current severe-aware estimator is not functionally useful as implemented, but it also produced enough evidence to avoid the wrong next step. Manual review did not localize the main failure to glomerulus/non-glomerulus mask quality. A read-only subject-heldout diagnostic showed recoverable adjudicated severe signal in existing morphology features when the objective is high-sensitivity severe detection rather than conservative ordinal/scalar burden regression: morphology-only balanced logistic screening reached AUROC `0.829`, average precision `0.326`, recall `0.676` at threshold `0.5`, recall `0.803` near threshold `0.477`, and recall `0.944` at threshold `0.3`. The current-data problem is therefore not proven impossible; P2 partly failed because it used the wrong operational objective and thresholding for severe detection. The next change should run an autonomous, evidence-gated modeling loop whose main product priority is MR TIFF deployment with a defensible endotheliosis grade quantification model. Severe grade 2/3 sensitivity is a required safety and utility gate for that product, not the product by itself.

## What Changes

- Add a grade-model workflow whose primary deliverable is a deployable MR TIFF segmentation-to-quantification path with the best supportable current-data grade output.
- Use all current labeled scored/mask-paired data for grouped development validation, model selection, and final refit rather than holding out a scarce internal locked test set.
- Evaluate ordinal outputs as ordered rubric predictions and coarser deployable bands, rather than assuming scalar burden is required.
- Evaluate high-sensitivity severe-risk behavior as a safety gate inside the grade model, with threshold selection inside grouped development folds rather than fixed default `0.5` thresholds.
- Treat multi-glomerulus image labels as image-level aggregate supervision, not per-glomerulus truth. P3 may test aggregate-aware multi-instance features and image-level aggregation without requiring recropping or assigning new per-glomerulus scores.
- Run an autonomous feature/model/error loop over existing ROI/QC, deterministic morphology, learned ROI, reduced embedding, learned/embedding-heavy neural-feature candidates, and new severe-focused feature transforms if they can be generated from existing ROI images and masks.
- Require a learned/embedding-heavy candidate lane that tests neural-network-derived ROI embeddings/features under grouped development validation rather than excluding them a priori.
- Write complete runtime evidence for each loop iteration: candidate metrics, threshold curves, test-set behavior, false negatives, false positives, ordinal confusion, feature ablations, review HTML, and a final product verdict.
- Produce explicit final states: README-facing deployable MR TIFF grade model, model-ready current-data grade model pending MR deployment, diagnostic-only model, or current-data insufficient after bounded attempts.
- Require an end-to-end MR TIFF deployment test before any README-facing deployable claim: supported segmentation artifact, whole-field TIFF tiling, glomerulus segmentation, component filtering, ROI extraction, P3 feature generation, final quantification inference, image-level aggregation, and human-vs-inferred concordance when MR labels are available.
- Keep claims current-data and source-sensitive unless an explicit test set and source-support evidence justify stronger wording.

## Capabilities

### New Capabilities

- `endotheliosis-grade-model`: Defines the autonomous P3 workflow for turning current ROI/mask/score/adjudication artifacts into the best supportable severe-risk or ordinal deployed quantification product, or into a clear insufficiency verdict.

### Modified Capabilities

- `severe-aware-ordinal-endotheliosis-estimator`: P3 consumes P2 artifacts and adjudications but does not mutate the archived P2 evidence.
- `ordinal-quantification-stability`: P3 reuses the six-bin rubric and tests ordinal/banded outputs under finite grouped out-of-fold evaluation.
- `morphology-aware-quantification-features`: P3 may use morphology features as predictive covariates, but it must not claim they are validated biological closed-lumen measurements unless existing readiness gates pass.
- `endotheliosis-burden-index`: P3 may report ordinal or banded burden outputs, but scalar burden remains non-reportable unless gates pass.

## Explicit Decisions

- Change ID: `p3-functional-severe-ordinal-quantification`.
- Primary runtime output root: `burden_model/endotheliosis_grade_model/`.
- Primary module target: `src/eq/quantification/endotheliosis_grade_model.py`.
- Pipeline integration target: `src/eq/quantification/pipeline.py`, after P2 severe-aware artifacts are available.
- Primary product target: MR TIFF deployment of a defensible endotheliosis grade quantification model.
- Primary supervised label target for model fitting: image-level grade/ordinal score, with adjudicated `score >= 2` severe status used as the severe safety gate when `severe_false_negative_adjudications.json` exists.
- Reviewed rubric label overrides may be supplied explicitly through `configs/endotheliosis_quantification.yaml` at `inputs.label_overrides`. These overrides replace row-level `score` values only for listed `subject_image_id` rows and must write an audit under `scored_examples/`; they must not infer labels for unreviewed rows.
- Label unit: available scores are image-level scores. When an image contains multiple glomeruli, the score is treated as an image-level aggregate label and not as a guaranteed per-glomerulus label.
- ROI row identity for learned ROI, morphology, and P3 optional feature joins is `subject_image_id` plus `glomerulus_id`. Row-level feature tables must preserve `glomerulus_id` when available and must not silently fall back to image-level joins when multiple ROI/glomerulus rows share a `subject_image_id`.
- Secondary targets:
  - original six-bin ordinal score `[0, 0.5, 1, 1.5, 2, 3]`
  - deployable ordinal bands: `none_low=[0,0.5]`, `mild_mod=[1,1.5]`, `severe=[2,3]`
  - optional four-band output if supported: `0`, `0.5`, `1/1.5`, `2/3`
- No internal locked test set is required by P3. Current labeled scored/mask-paired data should be used for grouped development validation and then final refit on all eligible labeled rows.
- Threshold selection, feature selection, dimensionality reduction, calibration, and model selection must happen inside grouped development folds. The final refit may use all eligible labeled rows only after candidate selection is complete.
- Severe-risk success gate:
  - preferred: grouped development out-of-fold severe recall `>= 0.80`, precision `>= 0.25`, finite outputs, no subject leakage, plus MR deployment test completion
  - minimum usable triage: grouped development out-of-fold severe recall `>= 0.90` with precision `>= 0.15`, plus MR deployment test completion and an explicit high-sensitivity review/triage claim boundary
- Ordinal/banded success gate:
  - deployable three-band grouped development out-of-fold balanced accuracy `>= 0.50`, severe-band recall `>= 0.80`, adjacent-band error dominance, plus MR deployment test completion
  - six-bin exact-score output is reportable only if it beats naive/majority and adjacent-baseline comparators with finite calibrated probabilities
- If gates conflict, README-facing deployment requires the output that is most defensible on MR TIFF deployment while preserving the severe safety gate. A severe-only triage output is acceptable only when richer grade outputs fail but severe-risk behavior and MR deployment pass.
- P3 SHALL test whether multi-component/aggregate-aware features improve performance: component count, component size distribution, component-level feature summaries, high-risk component maximum, mean/median, upper quantiles, and disagreement/spread. These are allowed because they use existing masks and images without new manual labels.
- P3 SHALL NOT invent per-glomerulus labels from an image-average score. Per-glomerulus predictions may be diagnostic only unless they are aggregated back to the image-level target or externally labeled later.
- If all candidates fail, the workflow must write `current_data_insufficient` with the strongest failed candidate, failure evidence, and the minimum additional data likely needed.
- No new manual labels are required during the apply pass. Existing adjudications are used as input. Review artifacts are generated for later human inspection, not as blockers for unattended execution.
- Segmentation-backbone comparison is out of scope unless P3 evidence shows feature failures are caused by mask geometry rather than severity modeling.
- Learned ROI and embedding-heavy features are required P3 candidate inputs. They may be selected if they improve grouped development out-of-fold severe or ordinal-band gates without leakage, nonfinite outputs, or unacceptable source/cohort sensitivity; otherwise they remain documented failed candidates rather than silently disappearing.
- Repeated grouped-validation, threshold-selection, warning-capture, artifact-manifest, finite-feature, and deployment-gating logic must be governed by shared quantification helpers rather than reimplemented as ad hoc P3-only rules.
- Candidate family names must be source-truthful: embedding candidates require actual embedding columns, learned candidates require actual learned-feature columns, and README-facing candidates require an inference schema computable from the MR TIFF deployment path.
- A final model may be README-facing only if both the grouped development quantification gate and the MR TIFF deployment test pass.
- The MR TIFF deployment test SHALL use the existing `vegfri_mr` whole-field TIFF cohort under the runtime cohort manifest when available.
- Before MR TIFF deployment, P3 SHALL run a smaller Dox scored-no-mask smoke stage when the master manifest marks clean candidates with `eligible_dox_scored_no_mask_smoke = true`. This stage uses runtime-local copies of resolved Label Studio upload images with human grades but no manual masks to test segmentation-to-quantification in the familiar Dox domain.
- When Dox smoke fails because the selected severe model overcalls rather than misses severe cases, P3 SHALL write a compact overcall triage queue and HTML review report from the Dox predictions so human review is representative and bounded.
- If reviewed Dox cluster-representative false positives confirm usable non-severe overcalls, P3 SHALL reject the selected severe-risk candidate as diagnostic-only, remove final deployable model artifacts, and stop before MR TIFF deployment.
- If no supported current-namespace segmentation artifact or no MR TIFF input is available, P3 SHALL stop at `model_ready_pending_mr_tiff_deployment_smoke` or a lower verdict, not README-facing deployment.
- The deployment test target is segmentation-to-quantification usability and MR concordance, not segmentation model retraining or segmentation-backbone comparison.

## logging-contract

P3 participates in the existing `endotheliosis_quantification` execution surface as high-level function events only. It SHALL NOT create a new durable log root, attach independent file handlers, or implement subprocess teeing. Durable command capture remains owned by `eq run-config --config configs/endotheliosis_quantification.yaml` and the repo-wide execution logging contract.

## docs-impact

P3 will update runtime review/index surfaces and result summaries only if implementation produces a final verdict artifact. Public docs must remain current-state and verdict-scoped: deployable MR TIFF grade model, deployable severe triage fallback, diagnostic-only, or current-data-insufficient. No docs may claim scalar burden, closed-capillary percent, causal evidence, or external validation unless P3 gates explicitly support that scope.

README-facing wording is allowed only for `readme_facing_deployable_mr_tiff_grade_model`. A severe or ordinal model that passes grouped development quantification gates but lacks the MR TIFF deployment test may be documented in runtime outputs as model-ready, but not promoted in the README as a complete deployable segmentation-plus-quantification workflow.

## Impact

- Adds a contained grade-model runtime subtree under `burden_model/endotheliosis_grade_model/`.
- Reuses and, when rerun, updates existing `burden_model/primary_burden_index/`, `burden_model/learned_roi/`, `burden_model/source_aware_estimator/`, and `burden_model/severe_aware_ordinal_estimator/` artifacts in their current directories.
- Adds first-class model-family subtrees for newly fit grade-model families, with summary, diagnostics, predictions, model artifacts, evidence, and internal logs. Expected new subtrees are `three_band_ordinal_model/`, `four_band_ordinal_model/` when supported, `severe_triage_model/`, `aggregate_grade_model/`, and `embedding_grade_model/`.
- Uses `burden_model/endotheliosis_grade_model/` as the final selector and MR TIFF deployment layer. It records candidate coverage, final model selection, selected-model metadata, MR TIFF deployment evidence, selector diagnostics, and the executive summary.
- Adds `dox_scored_only_resolution_audit.csv` and `dox_scored_no_mask_smoke_manifest.csv` under `raw_data/cohorts/vegfri_dox/metadata/`, copies clean images under `raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images/`, and records Dox smoke eligibility/status columns in the master manifest while using canonical `image_path` for the localized smoke image.
- Adds Dox overcall triage outputs under `burden_model/endotheliosis_grade_model/deployment/` when the Dox smoke exposes excessive severe false positives.
- Adds a reviewed Dox overcall diagnostic that can downgrade a model-ready severe-risk candidate to diagnostic-only and remove final model artifacts when representative review confirms non-severe usable overcalls.
- Adds focused tests for grouped folds, threshold selection, no leakage, severe/ordinal gate logic, aggregate-label handling, manifest completeness, and insufficient-data verdict behavior.
- Adds shared quantification modeling contracts for common finite-matrix construction, recall-threshold selection, warning capture, JSON/manifest writing, and grouped-development metric labeling.
- May add severe-focused feature transforms derived from existing ROI image/mask crops.
- Does not retrain glomerulus segmentation models.
- Does not require new manual annotation during implementation.
- Does not claim external validation.
