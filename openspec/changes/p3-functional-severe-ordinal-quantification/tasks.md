## 1. Input Reconstruction And Feasibility Audit

- [ ] 1.1 Inspect P2 archived audit results, current P2 runtime outputs, primary burden morphology outputs, learned ROI outputs, source-aware outputs, and ordinal outputs.
- [ ] 1.2 Record current row, subject, cohort, score, original severe, and adjudicated severe support in `audit-results.md`.
- [ ] 1.3 Confirm whether P2 adjudication JSON exists; if yes, define adjudicated `score >= 2` as the primary severe target.
- [ ] 1.4 Confirm all candidate input rows have stable `subject_id`, `subject_image_id`, `cohort_id`, `score`, ROI image path, and ROI mask path or record hard blockers.
- [ ] 1.5 Decide final supported target ladder for this run: severe triage first, ordinal bands second, six-bin diagnostic third, scalar burden only if evidence unexpectedly supports it.
- [ ] 1.6 Record the post-P2 feasibility diagnostic in `audit-results.md`: morphology-only balanced logistic AUROC `0.829`, AP `0.326`, recall/precision/FN at thresholds `0.5`, `~0.477`, and `0.3`, and the conclusion that P2 did not prove the data are signal-free.

## 2. Grouped Development Validation

- [ ] 2.1 Implement deterministic subject-level grouped development fold generation under `burden_model/functional_quantification_p3/splits/`.
- [ ] 2.2 Stratify folds by subject-level maximum score, adjudicated severe-positive presence, cohort, and score-bin coverage where feasible.
- [ ] 2.3 Use all eligible labeled scored/mask-paired subjects in grouped development validation; do not reserve a fixed internal locked test set by default.
- [ ] 2.4 Write `splits/development_folds.csv` with subject, row, score, severe, and cohort counts by fold.
- [ ] 2.5 Add tests proving no subject appears in more than one development fold.
- [ ] 2.6 Add tests proving score coverage gaps are reported rather than hidden.
- [ ] 2.7 Add tests proving current-data metrics are labeled grouped out-of-fold estimates, not independent heldout estimates.

## 3. Feature Matrix And Diagnostics

- [ ] 3.1 Build `feature_sets/p3_feature_matrix.csv` by joining ROI/QC, morphology, learned ROI, reduced embedding, score, adjudication, and provenance columns.
- [ ] 3.2 Write `feature_sets/p3_feature_diagnostics.json` with missingness, nonfinite counts, rank diagnostics, near-zero variance, source predictability, and family availability.
- [ ] 3.3 Build learned/embedding-heavy feature views: learned ROI only, reduced embeddings, learned plus morphology, embedding plus morphology, and hybrid ROI/morphology/learned candidates.
- [ ] 3.4 Implement dimensionality control for embedding-heavy views using fold-internal PCA or feature selection; validation-fold labels must not affect component or feature choice.
- [ ] 3.5 Add robust standardized morphology transforms and explicitly record transform provenance.
- [ ] 3.6 Add bounded severe-focused interaction features among open-lumen, collapsed/slit-like, RBC, quality, and confounder morphology signals.
- [ ] 3.7 Keep deterministic morphology biological-readiness status attached to all morphology-derived feature families.
- [ ] 3.8 Add tests proving feature columns are finite after preprocessing or explicitly excluded.
- [ ] 3.9 Add tests proving learned/embedding dimensionality control is fitted without validation-fold label leakage.
- [ ] 3.10 Detect multi-component/multi-glomerulus inputs and write component counts, component area summaries, and aggregate-label diagnostics.
- [ ] 3.11 Build aggregate-aware image-level feature views using component-level mean, median, maximum, upper quantile, and spread summaries without assigning per-glomerulus labels.
- [ ] 3.12 Add tests proving image-average scores are not silently treated as true per-glomerulus labels.

## 4. Baseline And Comparator Models

- [ ] 4.1 Implement empirical-prior and majority baselines for severe status, three-band ordinal, four-band ordinal, and six-bin ordinal targets.
- [ ] 4.2 Include P2 selected severe-aware output as a baseline comparator where available.
- [ ] 4.3 Evaluate existing source-aware and learned ROI outputs as comparators without allowing them to leak validation-fold labels.
- [ ] 4.4 Write baseline metrics into `internal/candidate_metrics.csv`.

## 5. Severe-Risk Candidate Loop

- [ ] 5.1 Implement class-balanced logistic severe gates for ROI/QC, morphology, ROI/QC+morphology, and severe-focused derived features.
- [ ] 5.2 Implement regularization sweeps for logistic severe gates with grouped development validation.
- [ ] 5.3 Implement recall-targeted threshold selection inside grouped development folds for target recalls 0.80, 0.90, and 0.95.
- [ ] 5.4 Implement calibrated probability variants where calibration can be done inside development data without leakage.
- [ ] 5.5 Implement learned/embedding-heavy severe gates: `learned_roi_severe_gate`, `embedding_reduced_severe_gate`, `learned_morphology_severe_gate`, and `embedding_morphology_severe_gate`.
- [ ] 5.5A Implement aggregate-aware severe gates that use multi-component summary features and image-level severe labels.
- [ ] 5.6 Implement tree-based severe comparators with class weighting as exploratory candidates.
- [ ] 5.7 For every severe candidate, write AUROC, average precision, recall, precision, false negatives, false positives, threshold, fold metrics, warning status, finite-output status, and learned-feature source-sensitivity status where applicable.
- [ ] 5.8 Add tests proving threshold selection never uses validation-fold labels.
- [ ] 5.9 Add tests proving learned/embedding-heavy candidates can be selected when gates pass and are explicitly rejected when they fail source/overfit gates.
- [ ] 5.10 Add tests proving severe promotion gates distinguish deployable, high-sensitivity-triage, diagnostic-only, and failed states.

## 6. Ordinal And Banded Burden Loop

- [ ] 6.1 Implement three-band target encoding: `none_low=[0,0.5]`, `mild_mod=[1,1.5]`, `severe=[2,3]`.
- [ ] 6.2 Implement four-band target encoding: `0`, `0.5`, `1/1.5`, `2/3`, only when support diagnostics allow.
- [ ] 6.3 Evaluate stable multinomial or ordinal logistic candidates for three-band and four-band outputs.
- [ ] 6.4 Evaluate learned/embedding-heavy ordinal candidates: `learned_three_band_ordinal`, `embedding_three_band_ordinal`, `hybrid_three_band_ordinal`, and corresponding four-band variants when support allows.
- [ ] 6.4A Evaluate aggregate-aware ordinal-band candidates that summarize component-level features to the image-level target.
- [ ] 6.5 Evaluate six-bin ordinal comparator using the canonical supported rubric `[0, 0.5, 1, 1.5, 2, 3]`.
- [ ] 6.6 Report exact accuracy, balanced accuracy, adjacent accuracy, severe-band recall, non-adjacent error rate, confusion matrix, prediction-set behavior, and learned-feature source-sensitivity status where applicable.
- [ ] 6.7 Write `summary/ordinal_feasibility.json` with a clear decision: ordinal bands deployable, ordinal diagnostic only, or ordinal unsupported.
- [ ] 6.8 Add tests for ordinal band gate logic, learned/embedding ordinal candidate gating, and six-bin diagnostic downgrade behavior.

## 7. Autonomous Iteration And Stop Rules

- [ ] 7.1 Implement `internal/autonomous_loop_log.json` recording every candidate family, feature family, threshold target, gate result, and exclusion reason.
- [ ] 7.2 If severe recall fails but AUROC/AP suggest signal, automatically try lower threshold targets before declaring severe unsupported.
- [ ] 7.3 If severe recall passes but precision is low, classify as high-sensitivity review triage rather than asking for manual approval.
- [ ] 7.4 If three-band ordinal fails, automatically try four-band support checks and six-bin diagnostic output before declaring ordinal unsupported.
- [ ] 7.5 If learned/embedding features improve apparent performance but degrade grouped out-of-fold performance, exclude them and record overfit/source-sensitivity evidence.
- [ ] 7.6 If learned/embedding-heavy features improve grouped out-of-fold severe or ordinal-band gates without source/overfit failure, allow them to win final selection.
- [ ] 7.7 If all candidates fail, write `current_data_insufficient` with minimum additional-data recommendations.

## 8. Final Product Verdict And Quantification Artifact

- [ ] 8.1 Write `summary/final_product_verdict.json` and `.md` with one of `readme_facing_deployable_current_data_model`, `deployable_current_data_severe_triage`, `deployable_current_data_ordinal_bands`, `model_ready_pending_mr_tiff_deployment_smoke`, `diagnostic_only_current_data_model`, or `current_data_insufficient`.
- [ ] 8.2 Write `summary/model_selection_table.csv`, `summary/development_oof_metrics.csv`, and `summary/severe_threshold_selection.json`.
- [ ] 8.3 If gates pass, freeze the selected recipe and refit the final selected model on all eligible labeled scored/mask-paired data.
- [ ] 8.4 If gates pass, write `model/final_model.joblib`, `model/final_model_metadata.json`, `model/inference_schema.json`, and `model/deployment_smoke_predictions.csv`.
- [ ] 8.5 If gates fail, do not write deployable model artifacts; write diagnostics and strongest-failed-candidate evidence.
- [ ] 8.6 Ensure final report states current-data/source-sensitive claim boundary and does not claim external validation.
- [ ] 8.7 Ensure final report states whether multi-glomerulus aggregate labels helped, hurt, or remained an unresolved limitation.

## 8A. MR TIFF Deployment Test

- [ ] 8A.1 Identify the supported current-namespace glomerulus segmentation artifact that the MR deployment test will use, or record a hard blocker if none is available.
- [ ] 8A.2 Use the runtime `vegfri_mr` whole-field TIFF cohort from `raw_data/cohorts/manifest.csv` as the MR deployment-test input when available.
- [ ] 8A.3 Implement a deployment path that tiles whole-field TIFFs, runs glomerulus segmentation or a supported segmentation-output loading path, merges tile predictions into whole-field coordinates, filters components by area/quality, extracts accepted ROI image/mask records, computes the selected P3 inference feature schema, loads `model/final_model.joblib`, writes ROI-level predictions, and aggregates to image-level medians/summaries.
- [ ] 8A.4 Write `deployment/mr_tiff_smoke_manifest.csv`, `deployment/mr_tiff_smoke_predictions.csv`, `deployment/mr_tiff_smoke_report.html`, and `deployment/segmentation_quantification_contract.json`.
- [ ] 8A.5 When MR workbook human image-level medians or replicate summaries are available, write human-vs-inferred concordance metrics and plots; when labels are absent, mark the result as technical smoke only.
- [ ] 8A.6 Promote to `readme_facing_deployable_current_data_model` only if quantification gates pass and the MR TIFF deployment test passes.
- [ ] 8A.7 If quantification gates pass but MR deployment is blocked or fails, set `model_ready_pending_mr_tiff_deployment_smoke` and block README-facing deployment language.
- [ ] 8A.8 Add tests proving feature-table-only predictions cannot satisfy the README-facing deployment gate.

## 9. Review And Evidence Artifacts

- [ ] 9.1 Write `evidence/error_review.html` covering selected-product errors and representative successes.
- [ ] 9.2 Write `evidence/severe_false_negative_review.html` for grouped out-of-fold development severe false negatives.
- [ ] 9.3 Write `evidence/ordinal_confusion_review.html` for adjacent, non-adjacent, severe-band, and correct ordinal examples.
- [ ] 9.4 Include raw ROI, mask, observed score, adjudicated severe label, predicted severe risk, predicted band/score, fold/split, cohort, and subject in review panels.
- [ ] 9.5 Write `INDEX.md` explaining what product was selected, what to open first, what not to claim, and what failed if no product was selected.
- [ ] 9.6 Write `summary/executive_summary.md` with what was performed, key results, selected or failed product status, MR deployment-test result, limitations, and concrete next steps.

## 10. Pipeline Integration

- [ ] 10.1 Add `evaluate_functional_p3_quantification` in `src/eq/quantification/functional_p3.py`.
- [ ] 10.2 Export the evaluator from `src/eq/quantification/__init__.py`.
- [ ] 10.3 Call P3 from `src/eq/quantification/pipeline.py` after P2 severe-aware artifacts are available.
- [ ] 10.4 Update combined quantification review links to include `functional_quantification_p3/INDEX.md` and final product status.
- [ ] 10.5 Update result summaries only with verdict-scoped language.
- [ ] 10.6 Update README-facing snippets only when the final verdict is `readme_facing_deployable_current_data_model`; otherwise keep deployment language in runtime reports only.

## 11. Tests

- [ ] 11.1 Add unit coverage for input reconstruction and adjudication target construction.
- [ ] 11.2 Add unit coverage for grouped fold determinism, score coverage reporting, and no leakage.
- [ ] 11.3 Add unit coverage for feature diagnostics and finite preprocessing.
- [ ] 11.4 Add unit coverage for severe threshold selection and severe gate verdict states.
- [ ] 11.5 Add unit coverage for ordinal band gate verdict states.
- [ ] 11.6 Add unit coverage for manifest completeness and no unmanifested P3 artifacts.
- [ ] 11.7 Add focused pipeline integration coverage for P3 review links and result-summary verdict rows.
- [ ] 11.8 Add regression coverage proving failed gates do not write deployable model artifacts.
- [ ] 11.9 Add regression coverage for MR TIFF deployment-test gating and README-facing verdict blocking.
- [ ] 11.10 Add regression coverage for aggregate-label handling and executive-summary completeness.

## 12. Runtime Execution

- [ ] 12.1 Run focused P3 unit tests.
- [ ] 12.2 Run focused quantification tests covering primary burden, learned ROI, source-aware, severe-aware, P3, and pipeline integration.
- [ ] 12.3 Run the full quantification workflow with `eq run-config --config configs/endotheliosis_quantification.yaml`.
- [ ] 12.4 Inspect P3 final verdict, grouped development folds, out-of-fold metrics, severe false negatives, ordinal feasibility, model artifact status, MR deployment artifacts, and manifest.
- [ ] 12.5 If quantification gates pass, run the MR TIFF deployment test and inspect the inference schema plus segmentation-quantification contract.
- [ ] 12.6 If the final verdict is diagnostic-only or insufficient, inspect the strongest failed candidate and minimum additional-data recommendation.
- [ ] 12.7 Inspect `summary/executive_summary.md` and confirm it answers what was performed, what the results were, and what is next.

## 13. Validation

- [ ] 13.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_functional_p3.py`.
- [ ] 13.2 Run focused quantification test suite.
- [ ] 13.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/quantification tests/unit`.
- [ ] 13.4 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff format --check src/eq/quantification tests/unit`.
- [ ] 13.5 Run `openspec validate p3-functional-severe-ordinal-quantification --strict`.
- [ ] 13.6 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p3-functional-severe-ordinal-quantification`.
- [ ] 13.7 Record final validation results, product verdict, and residual risks in `audit-results.md`.
