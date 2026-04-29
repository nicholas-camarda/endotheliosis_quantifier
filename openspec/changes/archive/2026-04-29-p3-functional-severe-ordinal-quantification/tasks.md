## 1. Input Reconstruction And Feasibility Audit

- [x] 1.1 Inspect P2 archived audit results, current P2 runtime outputs, primary burden morphology outputs, learned ROI outputs, source-aware outputs, and ordinal outputs.
- [x] 1.2 Record current row, subject, cohort, score, original severe, and adjudicated severe support in `audit-results.md`.
- [x] 1.3 Confirm whether P2 adjudication JSON exists; if yes, define adjudicated `score >= 2` as the primary severe target.
- [x] 1.4 Confirm all candidate input rows have stable `subject_id`, `subject_image_id`, `cohort_id`, `score`, ROI image path, and ROI mask path or record hard blockers.
- [x] 1.5 Decide final supported target ladder for this run: MR TIFF deployable grade model first, MR TIFF severe-risk triage fallback second, diagnostic-only third, with scalar/score-like burden reportable only if evidence unexpectedly supports it.
- [x] 1.6 Record the post-P2 feasibility diagnostic in `audit-results.md`: morphology-only balanced logistic AUROC `0.829`, AP `0.326`, recall/precision/FN at thresholds `0.5`, `~0.477`, and `0.3`, and the conclusion that P2 did not prove the data are signal-free.

## 2. Grouped Development Validation

- [x] 2.1 Implement deterministic subject-level grouped development fold generation under `burden_model/endotheliosis_grade_model/splits/`.
- [x] 2.2 Stratify folds by subject-level maximum score, adjudicated severe-positive presence, cohort, and score-bin coverage where feasible.
- [x] 2.3 Use all eligible labeled scored/mask-paired subjects in grouped development validation; do not reserve a fixed internal locked test set by default.
- [x] 2.4 Write `splits/development_folds.csv` with subject, row, score, severe, and cohort counts by fold.
- [x] 2.5 Add tests proving no subject appears in more than one development fold.
- [x] 2.6 Add tests proving score coverage gaps are reported rather than hidden.
- [x] 2.7 Add tests proving current-data metrics are labeled grouped out-of-fold estimates, not independent heldout estimates.

## 3. Feature Matrix And Diagnostics

- [x] 3.1 Build an in-memory grade-model feature matrix by joining ROI/QC, morphology, learned ROI, reduced embedding, score, adjudication, and provenance columns from existing burden-model artifacts.
- [x] 3.2 Write compact diagnostics under `internal/feature_diagnostics.json` with missingness, nonfinite counts, rank diagnostics, near-zero variance, source predictability, family availability, and upstream artifact references.
- [x] 3.3 Build learned/embedding-heavy feature views: learned ROI only, reduced embeddings, learned plus morphology, embedding plus morphology, and hybrid ROI/morphology/learned candidates.
- [x] 3.4 Implement dimensionality control for embedding-heavy views using fold-internal PCA or feature selection; validation-fold labels must not affect component or feature choice.
- [x] 3.5 Add robust standardized morphology transforms and explicitly record transform provenance.
- [x] 3.6 Add bounded severe-focused interaction features among open-lumen, collapsed/slit-like, RBC, quality, and confounder morphology signals.
- [x] 3.7 Keep deterministic morphology biological-readiness status attached to all morphology-derived feature families.
- [x] 3.8 Add tests proving feature columns are finite after preprocessing or explicitly excluded.
- [ ] 3.9 Add tests proving learned/embedding dimensionality control is fitted without validation-fold label leakage.
- [ ] 3.10 Detect multi-component/multi-glomerulus inputs and write component counts, component area summaries, and aggregate-label diagnostics.
- [ ] 3.11 Build aggregate-aware image-level feature views using component-level mean, median, maximum, upper quantile, and spread summaries without assigning per-glomerulus labels.
- [ ] 3.12 Add tests proving image-average scores are not silently treated as true per-glomerulus labels.

## 4. Baseline And Comparator Models

- [x] 4.1 Implement empirical-prior and majority baselines for severe status, three-band ordinal, four-band ordinal, and six-bin ordinal targets.
- [ ] 4.2 Include P2 selected severe-aware output as a baseline comparator where available.
- [ ] 4.3 Evaluate existing source-aware and learned ROI outputs as comparators without allowing them to leak validation-fold labels.
- [x] 4.4 Write baseline metrics into `internal/candidate_metrics.csv`.

## 5. Severe-Risk Candidate Loop

- [x] 5.1 Implement class-balanced logistic severe gates for ROI/QC, morphology, ROI/QC+morphology, and severe-focused derived features.
- [x] 5.2 Implement regularization sweeps for logistic severe gates with grouped development validation.
- [x] 5.3 Implement recall-targeted threshold selection inside grouped development folds for target recalls 0.80, 0.90, and 0.95.
- [ ] 5.4 Implement calibrated probability variants where calibration can be done inside development data without leakage.
- [x] 5.5 Implement learned/embedding-heavy severe gates: `learned_roi_severe_gate`, `embedding_reduced_severe_gate`, `learned_morphology_severe_gate`, and `embedding_morphology_severe_gate`.
- [x] 5.5A Implement aggregate-aware severe gates that use multi-component summary features and image-level severe labels.
- [x] 5.6 Implement tree-based severe comparators with class weighting as exploratory candidates.
- [x] 5.7 For every severe candidate, write AUROC, average precision, recall, precision, false negatives, false positives, threshold, fold metrics, warning status, finite-output status, and learned-feature source-sensitivity status where applicable.
- [ ] 5.8 Add tests proving threshold selection never uses validation-fold labels.
- [ ] 5.9 Add tests proving learned/embedding-heavy candidates can be selected when gates pass and are explicitly rejected when they fail source/overfit gates.
- [ ] 5.10 Add tests proving severe promotion gates distinguish deployable, high-sensitivity-triage, diagnostic-only, and failed states.

## 6. Ordinal And Banded Burden Loop

- [x] 6.1 Implement three-band target encoding: `none_low=[0,0.5]`, `mild_mod=[1,1.5]`, `severe=[2,3]`.
- [x] 6.2 Implement four-band target encoding: `0`, `0.5`, `1/1.5`, `2/3`, only when support diagnostics allow.
- [x] 6.3 Evaluate stable multinomial or ordinal logistic candidates for three-band and four-band outputs.
- [x] 6.4 Evaluate learned/embedding-heavy ordinal candidates: `learned_three_band_ordinal`, `embedding_three_band_ordinal`, `hybrid_three_band_ordinal`, and corresponding four-band variants when support allows.
- [x] 6.4A Evaluate aggregate-aware ordinal-band candidates that summarize component-level features to the image-level target.
- [x] 6.5 Evaluate six-bin ordinal comparator using the canonical supported rubric `[0, 0.5, 1, 1.5, 2, 3]`.
- [ ] 6.6 Report exact accuracy, balanced accuracy, adjacent accuracy, severe-band recall, non-adjacent error rate, confusion matrix, prediction-set behavior, and learned-feature source-sensitivity status where applicable.
- [x] 6.7 Write `summary/ordinal_feasibility.json` with a clear decision: ordinal bands deployable, ordinal diagnostic only, or ordinal unsupported.
- [ ] 6.8 Add tests for ordinal band gate logic, learned/embedding ordinal candidate gating, and six-bin diagnostic downgrade behavior.

## 6A. First-Class Model-Family Output Subtrees

- [x] 6A.1 Write three-band ordinal model outputs under `burden_model/three_band_ordinal_model/` with `INDEX.md`, `summary/`, `diagnostics/`, `predictions/`, `model/`, `evidence/`, and `internal/`.
- [x] 6A.2 Write four-band ordinal model outputs under `burden_model/four_band_ordinal_model/` when score support allows; if unsupported, record the support blocker in selector coverage and diagnostics.
- [x] 6A.3 Write severe triage outputs under `burden_model/severe_triage_model/` with high-sensitivity threshold diagnostics and error evidence.
- [x] 6A.4 Write aggregate-aware grade outputs under `burden_model/aggregate_grade_model/` with aggregate-label diagnostics and image-level prediction evidence.
- [x] 6A.5 Write embedding-heavy grade outputs under `burden_model/embedding_grade_model/` with embedding source-predictability and source-sensitivity diagnostics.
- [x] 6A.6 For every first-class model-family subtree, write `diagnostics/input_support.json`, `diagnostics/feature_diagnostics.json`, `diagnostics/fold_diagnostics.json`, `diagnostics/source_sensitivity.json`, and `diagnostics/gate_diagnostics.json`.
- [ ] 6A.7 Add family-specific diagnostics where applicable: `embedding_source_predictability.json`, `aggregate_label_diagnostics.json`, `calibration_diagnostics.json`, `threshold_selection_diagnostics.json`, `mr_tiff_deployment_diagnostics.json`, and `hard_blockers.json`.
- [x] 6A.8 Add tests proving no required newly fit model family is represented only by `burden_model/endotheliosis_grade_model/internal/`.

## 7. Autonomous Iteration And Stop Rules

- [x] 7.1 Implement `internal/autonomous_loop_log.json` recording every candidate family, feature family, threshold target, gate result, and exclusion reason.
- [x] 7.2 If severe recall fails but AUROC/AP suggest signal, automatically try lower threshold targets before declaring severe unsupported.
- [x] 7.3 If severe recall passes but precision is low, classify as high-sensitivity review triage rather than asking for manual approval.
- [x] 7.4 If three-band ordinal fails, automatically try four-band support checks and six-bin diagnostic output before declaring ordinal unsupported.
- [ ] 7.5 If learned/embedding features improve apparent performance but degrade grouped out-of-fold performance, exclude them and record overfit/source-sensitivity evidence.
- [x] 7.6 If learned/embedding-heavy features improve grouped out-of-fold severe or ordinal-band gates without source/overfit failure, allow them to win final selection.
- [x] 7.7 If all candidates fail, write `current_data_insufficient` with minimum additional-data recommendations.

## 8. Final Product Verdict And Quantification Artifact

- [x] 8.1 Write `summary/final_product_verdict.json` and `.md` with one of `readme_facing_deployable_mr_tiff_grade_model`, `readme_facing_deployable_mr_tiff_severe_triage`, `model_ready_pending_mr_tiff_deployment_smoke`, `diagnostic_only_current_data_model`, or `current_data_insufficient`.
- [x] 8.2 Write `summary/candidate_coverage_matrix.csv`, `summary/model_selection_table.csv`, `summary/development_oof_metrics.csv`, and `summary/severe_threshold_selection.json`.
- [x] 8.3 If gates pass, freeze the selected recipe and refit the final selected model on all eligible labeled scored/mask-paired data.
- [x] 8.4 If gates pass, write `model/final_model.joblib`, `model/final_model_metadata.json`, `model/inference_schema.json`, and `model/deployment_smoke_predictions.csv`; metadata must link the selected model to its source family subtree, diagnostics, predictions, and metrics.
- [x] 8.5 If gates fail, do not write deployable model artifacts; write diagnostics and strongest-failed-candidate evidence.
- [x] 8.6 Ensure final report states current-data/source-sensitive claim boundary and does not claim external validation.
- [ ] 8.7 Ensure final report states whether multi-glomerulus aggregate labels helped, hurt, or remained an unresolved limitation.
- [x] 8.8 Write `diagnostics/selector_diagnostics.json` and `diagnostics/candidate_family_gate_diagnostics.json` under `burden_model/endotheliosis_grade_model/`.

## 8A. MR TIFF Deployment Test

- [x] 8A.1 Identify the supported current-namespace glomerulus segmentation artifact that the MR deployment test will use, or record a hard blocker if none is available.
- [ ] 8A.2 Use the runtime `vegfri_mr` whole-field TIFF cohort from `raw_data/cohorts/manifest.csv` as the MR deployment-test input when available.
- [ ] 8A.3 Implement a deployment path that tiles whole-field TIFFs, runs glomerulus segmentation or a supported segmentation-output loading path, merges tile predictions into whole-field coordinates, filters components by area/quality, extracts accepted ROI image/mask records, computes the selected P3 inference feature schema, loads `model/final_model.joblib`, writes ROI-level predictions, and aggregates to image-level medians/summaries.
- [ ] 8A.4 Write `deployment/mr_tiff_smoke_manifest.csv`, `deployment/mr_tiff_smoke_predictions.csv`, `deployment/mr_tiff_smoke_report.html`, and `deployment/segmentation_quantification_contract.json`.
- [ ] 8A.5 When MR workbook human image-level medians or replicate summaries are available, write human-vs-inferred concordance metrics and plots; when labels are absent, mark the result as technical smoke only.
- [x] 8A.6 Promote to a README-facing MR TIFF verdict only if quantification gates, severe safety gates, and the MR TIFF deployment test pass.
- [x] 8A.7 If quantification gates pass but MR deployment is blocked or fails, set `model_ready_pending_mr_tiff_deployment_smoke` and block README-facing deployment language.
- [x] 8A.8 Add tests proving feature-table-only predictions cannot satisfy the README-facing deployment gate.

## 9. Review And Evidence Artifacts

- [x] 9.1 Write `evidence/error_review.html` covering selected-product errors and representative successes.
- [x] 9.2 Write `evidence/severe_false_negative_review.html` for grouped out-of-fold development severe false negatives.
- [x] 9.3 Write `evidence/ordinal_confusion_review.html` for adjacent, non-adjacent, severe-band, and correct ordinal examples.
- [ ] 9.4 Include raw ROI, mask, observed score, adjudicated severe label, predicted severe risk, predicted band/score, fold/split, cohort, and subject in review panels.
- [x] 9.5 Write `INDEX.md` explaining what product was selected, what to open first, what not to claim, and what failed if no product was selected.
- [x] 9.6 Write `summary/executive_summary.md` with what was performed, key results, selected or failed product status, MR deployment-test result, limitations, and concrete next steps.
- [x] 9.7 Write `summary/input_artifact_index.json` listing every consumed upstream burden-model artifact, its existing path, role, and whether it was copied, linked, or read-only.
- [x] 9.8 Ensure `summary/candidate_coverage_matrix.csv` records every committed candidate family with `family_id`, `subtree_path`, `required_or_exploratory`, `run_status`, `candidate_ids`, `metrics_path`, `diagnostics_path`, `predictions_path`, `gate_status`, `selected`, and `failure_or_exclusion_reason`.

## 10. Pipeline Integration

- [x] 10.1 Add `evaluate_endotheliosis_grade_model` in `src/eq/quantification/endotheliosis_grade_model.py`.
- [x] 10.2 Export the evaluator from `src/eq/quantification/__init__.py`.
- [x] 10.3 Call P3 from `src/eq/quantification/pipeline.py` after P2 severe-aware artifacts are available.
- [x] 10.4 Update combined quantification review links to include `endotheliosis_grade_model/INDEX.md` and final product status.
- [x] 10.5 Update result summaries only with verdict-scoped language.
- [x] 10.6 Update README-facing snippets only when the final verdict is `readme_facing_deployable_mr_tiff_grade_model` or `readme_facing_deployable_mr_tiff_severe_triage`; otherwise keep deployment language in runtime reports only.

## 11. Tests

- [x] 11.1 Add unit coverage for input reconstruction and adjudication target construction.
- [x] 11.2 Add unit coverage for grouped fold determinism, score coverage reporting, and no leakage.
- [x] 11.3 Add unit coverage for feature diagnostics and finite preprocessing.
- [ ] 11.4 Add unit coverage for severe threshold selection and severe gate verdict states.
- [ ] 11.5 Add unit coverage for ordinal band gate verdict states.
- [ ] 11.6 Add unit coverage for manifest completeness and no unmanifested P3 artifacts.
- [x] 11.7 Add focused pipeline integration coverage for P3 review links and result-summary verdict rows.
- [x] 11.8 Add regression coverage proving failed gates do not write deployable model artifacts.
- [x] 11.9 Add regression coverage for MR TIFF deployment-test gating and README-facing verdict blocking.
- [ ] 11.10 Add regression coverage for aggregate-label handling and executive-summary completeness.
- [x] 11.11 Add regression coverage proving the grade-model subtree does not create duplicate `feature_sets/`, `calibration/`, `validation/`, `summaries/`, or candidate-screen roots when existing burden-model artifacts can be referenced.
- [x] 11.12 Add regression coverage proving every required first-class model family writes a source subtree with summary, diagnostics, predictions, model, evidence, and internal directories.
- [x] 11.13 Add regression coverage proving missing expected upstream artifacts for required families are hard failures and appear in `diagnostics/hard_blockers.json`.

## 12. Runtime Execution

- [x] 12.1 Run focused P3 unit tests.
- [x] 12.2 Run focused quantification tests covering primary burden, learned ROI, source-aware, severe-aware, P3, and pipeline integration.
- [x] 12.3 Run the full quantification workflow with `eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 12.4 Inspect P3 final verdict, grouped development folds, out-of-fold metrics, severe false negatives, ordinal feasibility, model artifact status, MR deployment artifacts, and manifest.
- [x] 12.5 If quantification gates pass, run the MR TIFF deployment test and inspect the inference schema plus segmentation-quantification contract.
- [x] 12.6 If the final verdict is diagnostic-only or insufficient, inspect the strongest failed candidate and minimum additional-data recommendation.
- [x] 12.7 Inspect `summary/executive_summary.md` and confirm it answers what was performed, what the results were, and what is next.

## 13. Validation

- [x] 13.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_endotheliosis_grade_model.py`.
- [x] 13.2 Run focused quantification test suite.
- [x] 13.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/quantification tests/unit`.
- [x] 13.4 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff format --check src/eq/quantification tests/unit`.
- [x] 13.5 Run `openspec validate p3-functional-severe-ordinal-quantification --strict`.
- [x] 13.6 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p3-functional-severe-ordinal-quantification`.
- [x] 13.7 Record final validation results, product verdict, and residual risks in `audit-results.md`.

## 14. Research-Partner Audit And Shared Governance

- [x] 14.1 Audit repo-wide quantification and segmentation surfaces for repeated grouped-validation, threshold-selection, finite-feature, warning-capture, artifact-manifest, and deployment-gating logic.
- [x] 14.2 Extend P3 proposal/design/spec artifacts with a central modeling-contract requirement.
- [x] 14.3 Add `src/eq/quantification/modeling_contracts.py` for shared finite-matrix, threshold-selection, warning-capture, JSON, manifest, and metric-label contracts.
- [x] 14.4 Refactor P3 selector code to use the shared modeling-contract helpers for newly touched mechanics.
- [x] 14.5 Prevent source-specific candidate family drift: learned candidates require learned columns, embedding candidates require embedding columns, and missing source-specific inputs are unavailable rather than silently replaced by morphology-only columns.
- [x] 14.6 Run the expanded bounded candidate loop with regularization sweeps and severe-focused interaction features.
- [x] 14.7 Record the pre-reviewed-rubric P3 diagnostic/no-product verdict when no deployable severe or ordinal candidate cleared gates.
- [x] 14.8 Ensure failed final gates remove stale deployable model artifacts from prior selector runs.

## 15. Reviewed Rubric Label Overrides

- [x] 15.1 Generate a grade atlas and prioritized review queues from current scored ROI artifacts without treating projected labels as ground truth.
- [x] 15.2 Convert completed reviewer rubric passes into `rubric_label_overrides_for_next_modeling_run.csv` keyed by `subject_image_id`.
- [x] 15.3 Add `inputs.label_overrides` to `configs/endotheliosis_quantification.yaml` pointing to the reviewed rubric override artifact.
- [x] 15.4 Apply reviewed rubric overrides before ROI extraction, embedding extraction, burden modeling, learned ROI modeling, source-aware modeling, severe-aware modeling, and P3 selection.
- [x] 15.5 Write `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json` when overrides are applied.
- [x] 15.6 Fail closed for missing override files, duplicate override rows, unrecognized scores, nonnumeric scores, and override rows that do not match scored examples.
- [x] 15.7 Add regression coverage proving optional feature-table merges cannot reintroduce stale score columns over caller-provided labels.
- [x] 15.8 Add regression coverage proving score label overrides replace scored-example labels and write an audit.
- [x] 15.9 Rerun `eq run-config --config configs/endotheliosis_quantification.yaml` with reviewed rubric overrides through the normal workflow.
- [x] 15.10 Inspect normal-workflow P3 verdict, final model artifacts, label-override audit, and MR deployment blocker state after the rerun.

## 16. Row-Level Optional Feature Join Contract

- [x] 16.1 Preserve `glomerulus_id` in `learned_roi/feature_sets/learned_roi_features.csv` when the embedding/ROI input rows provide it.
- [x] 16.2 Require `subject_image_id` plus `glomerulus_id` for P3 joins against row-level morphology and learned ROI feature tables.
- [x] 16.3 Treat missing required row-level join keys or duplicate optional-table join keys as hard blockers instead of silently falling back to image-level joins.
- [x] 16.4 Add regression coverage with two rows sharing `subject_image_id` but carrying different `glomerulus_id` values.
- [x] 16.5 Add regression coverage proving a learned ROI row-level table without `glomerulus_id` records an unusable-join hard blocker rather than crashing during normal P3 evaluation.

## 17. Dox Scored-No-Mask Pre-MR Smoke

- [x] 17.1 Add a reusable Dox scored-only resolution audit/export that writes `raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.csv`.
- [x] 17.2 Resolve each Dox `scored_only` row to exact Label Studio upload images, copy clean images to `raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images/`, and record source path, localized path, image hash, match count, duplicate-name flag, conflicting-score flag, missing-score flag, and clean-smoke status.
- [x] 17.3 Write `raw_data/cohorts/vegfri_dox/metadata/dox_scored_no_mask_smoke_manifest.csv` containing only nonduplicate, nonconflicting, nonmissing-score rows with exactly one resolved upload image and a localized runtime image copy.
- [x] 17.4 Add a CLI command `eq dox-scored-only-resolution-audit` for regenerating the audit and clean smoke manifest.
- [x] 17.5 Add regression coverage proving duplicates, conflicting scores, missing scores, and unresolved image rows are excluded from the clean smoke manifest but retained in the audit.
- [x] 17.6 Run the audit/export on the current runtime manifest and inspect row counts.
- [x] 17.6A Update the master manifest so clean Dox smoke rows use canonical `image_path` for the localized image, keep `mask_path` empty, and are governed by `eligible_dox_scored_no_mask_smoke` plus Dox smoke status columns.
- [x] 17.7 Implement the P3 Dox scored-no-mask smoke stage that runs supported segmentation, P3 feature generation, final model inference, image-level aggregation, and human-grade comparison on the clean smoke manifest.
- [x] 17.8 Write Dox smoke artifacts under `burden_model/endotheliosis_grade_model/deployment/`: `dox_scored_no_mask_smoke_manifest.csv`, `dox_scored_no_mask_smoke_predictions.csv`, `dox_scored_no_mask_smoke_summary.csv`, `dox_scored_no_mask_smoke_report.html`, and Dox-specific contract diagnostics.
- [x] 17.9 Require Dox scored-no-mask smoke to pass before MR TIFF deployment can be attempted, while keeping README-facing deployment blocked until MR TIFF smoke also passes.

## 18. Dox Overcall Active Triage

- [x] 18.1 Add `dox_scored_no_mask_smoke_threshold_curve.csv` so Dox smoke distinguishes threshold-choice failure from feature/model overcalling.
- [x] 18.2 Add a Dox overcall triage queue generated from deployment-computable prediction fields without rerunning segmentation.
- [x] 18.3 Include clustered false-positive representatives, highest-confidence false positives, threshold-boundary false positives, human severe references, and segmentation misses in the bounded review queue.
- [x] 18.4 Write `dox_scored_no_mask_overcall_triage_queue.csv` and `dox_scored_no_mask_overcall_triage_report.html` under `burden_model/endotheliosis_grade_model/deployment/`.
- [x] 18.5 Record the current Dox failure as severe-model overcalling after tiled MPS segmentation produced accepted ROIs for `210/212` clean smoke images.

## 19. Reviewed Dox Overcall Wrap-Up

- [x] 19.1 Preserve reviewer-entered Dox overcall triage annotations when regenerating the queue and report.
- [x] 19.2 Add a reviewed-overcall diagnostic that summarizes completed cluster-representative false-positive review rows.
- [x] 19.3 Treat reviewed usable non-severe cluster-representative overcalls as a hard rejection of the selected severe-risk candidate.
- [x] 19.4 Downgrade the current verdict to `diagnostic_only_current_data_model` when reviewed Dox overcalls are confirmed.
- [x] 19.5 Remove final deployable model artifacts after Dox review rejects the selected severe-risk candidate.
- [x] 19.6 Record the final P3 wrap-up decision and validation evidence in `audit-results.md`.
