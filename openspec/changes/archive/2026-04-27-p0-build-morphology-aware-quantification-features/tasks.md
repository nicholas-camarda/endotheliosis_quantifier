## 1. Output Contract Organization

- [x] 1.1 Add explicit burden output path mapping for `primary_model`, `validation`, `calibration`, `summaries`, `evidence`, `candidates`, `diagnostics`, and `feature_sets` under `src/eq/quantification/burden.py` or a local helper used by the quantification pipeline.
- [x] 1.2 Move future burden artifacts into the grouped subfolders without writing duplicate flat compatibility aliases.
- [x] 1.3 Update `quantification_review/quantification_review.html`, `results_summary.md`, `results_summary.csv`, and `readme_results_snippet.md` artifact links to the grouped paths.
- [x] 1.4 Update tests so candidate artifacts are asserted under `burden_model/candidates/` and serialized selected model artifacts are asserted under `burden_model/primary_model/`.

## 2. Morphology Feature Extraction

- [x] 2.1 Create `src/eq/quantification/morphology_features.py` with a public function that accepts the ROI table and writes one feature row per valid ROI.
- [x] 2.2 Implement open-lumen features: pale lumen area fraction, lumen candidate count, lumen area summary, circularity, eccentricity, and open-space density.
- [x] 2.3 Implement collapsed/slit/ridge features: ridge response, line density, skeleton length per mask area, slit-like object count, and ridge-to-lumen ratio.
- [x] 2.4 Implement erythrocyte-confounder features: RBC-like color burden, RBC-filled round-lumen candidates, RBC-filled lumen area fraction, and dark filled-lumen shape evidence.
- [x] 2.5 Implement quality/orientation features: blur/focus, stain or intensity range, orientation ambiguity, and lumen-detectability score.
- [x] 2.6 Write `burden_model/feature_sets/morphology_features.csv`, `morphology_feature_metadata.json`, and `subject_morphology_features.csv`.
- [x] 2.7 Write `burden_model/diagnostics/morphology_feature_diagnostics.json` with row count, subject count, feature count, nonfinite counts, zero and near-zero variance counts, missingness counts, and feature ranges.

## 3. Plug-And-Play Operator Review

- [x] 3.1 Create `src/eq/quantification/feature_review.py` to generate visual QA panels from morphology features and ROI crops.
- [x] 3.2 Write `burden_model/evidence/morphology_feature_review/feature_review.html` with raw ROI, mask, pale/open lumen overlay, RBC-filled lumen candidate overlay, collapsed/slit-like overlay, score, and key feature values.
- [x] 3.3 Write `feature_review_cases.csv` selecting high-score, low-score, high-uncertainty, high-RBC-confounder, high-collapsed-line, high-open-lumen, and poor-quality/orientation examples where available.
- [x] 3.4 Write `operator_adjudication_template.csv` with `case_id`, `subject_id`, `sample_id`, `image_id`, `score`, `open_empty_lumen_present`, `open_rbc_filled_lumen_present`, `collapsed_slit_like_lumen_present`, `poor_orientation_or_quality`, `feature_detection_problem`, `preferred_label_if_detection_wrong`, and `notes`.
- [x] 3.5 Add rerun logic that reads a completed adjudication CSV from the same review directory when present and writes an agreement summary without requiring a new CLI.
- [x] 3.6 Document the user workflow in `audit-results.md`: open HTML, fill template columns, save CSV, rerun the same YAML, inspect agreement summary.

## 4. Morphology Candidate Modeling

- [x] 4.1 Add image-level morphology candidate screens using subject-heldout folds and reporting stage-index MAE, grade-scale MAE, prediction-set coverage, average set size, cohort metrics, finite-output status, and warning status.
- [x] 4.2 Add subject-level morphology candidate screens by aggregating morphology features per `subject_id`, validating held-out subjects, and writing subject-level predictions.
- [x] 4.3 Write `burden_model/candidates/morphology_candidate_metrics.csv`, `subject_morphology_candidate_predictions.csv`, and `morphology_candidate_summary.json`.
- [x] 4.4 Compare morphology-only, morphology-plus-embedding, and current embedding/burden baselines without selecting candidates that emit nonfinite outputs.
- [x] 4.5 Record whether subject/cohort burden or per-image burden is selected for any README/docs-ready claim, or keep both exploratory if gates fail.

## 5. Report And Documentation Integration

- [x] 5.1 Add a morphology feature summary section to `quantification_review/quantification_review.html`.
- [x] 5.2 Link the feature QA report and candidate summary from the combined quantification review.
- [x] 5.3 Update `README.md`, `docs/ONBOARDING_GUIDE.md`, `docs/OUTPUT_STRUCTURE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md` after implementation so current docs reflect the grouped output layout and morphology-aware feature status.
- [x] 5.4 Ensure docs distinguish current implemented morphology features from future learned morphology encoders or mitochondria-transfer representation tests.

## 6. Tests And Validation

- [x] 6.1 Add unit tests for morphology feature schema, finite numeric outputs, bounds, and RBC-confounder fields.
- [x] 6.2 Add feature-review tests proving HTML, assets, review cases, operator template, and adjudication agreement artifacts are written.
- [x] 6.3 Add focused pipeline tests proving the grouped burden output layout is generated and old flat aliases are not written for new runs.
- [x] 6.4 Add candidate tests proving image-level candidates use subject-heldout folds and subject-level candidates aggregate by `subject_id`.
- [x] 6.5 Run changed-file lint and focused tests for quantification modules.
- [x] 6.6 Run the full test suite with `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 6.7 Run `openspec validate p0-build-morphology-aware-quantification-features --strict`.
- [x] 6.8 Run `python scripts/check_openspec_explicitness.py openspec/changes/p0-build-morphology-aware-quantification-features`.

## 7. Full Workflow Runtime Review

- [x] 7.1 Run the full workflow with `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 7.2 Inspect morphology feature distributions, diagnostics, QA panels, candidate metrics, subject-level predictions, and combined review outputs.
- [x] 7.3 Record in `audit-results.md` exact artifact links, what worked, what failed, whether RBC-filled patent-lumen cases were handled plausibly, and what the user should review next.
- [x] 7.4 If operator adjudication is not provided in the first run, record that status explicitly and leave clear instructions for the user's plug-and-play review pass.

## 8. Adjudication-Driven Slit Detector Revision

- [x] 8.1 Inspect completed operator adjudication and record dominant failure labels.
- [x] 8.2 Revise slit-like detector so elongated pale/dark slit candidates are not hidden inside open-lumen features.
- [x] 8.3 Revise overlay rendering so red slit candidates are drawn visibly above ridge/open/RBC overlays.
- [x] 8.4 Preserve the operator-reviewed case set on rerun instead of silently selecting a different review set.
- [x] 8.5 Add adjudication summary counts for completed rows, positive labels, and preferred failure labels.
- [x] 8.6 Rerun full workflow and record whether revised features detect user-labeled slit-positive cases.
- [x] 8.7 Record the remaining limitation that the revised detector improves slit visibility but appears over-sensitive and is not biology-ready.

## 9. Mesangial/Nuclear False-Slit Confounder Revision

- [x] 9.1 Amend morphology feature requirements to treat mesangial cells and compact nuclei as named false-slit confounders.
- [x] 9.2 Add nuclear/mesangial confounder feature columns to `src/eq/quantification/morphology_features.py`.
- [x] 9.3 Exclude nuclear/mesangial-like masks from collapsed/slit-like feature computation.
- [x] 9.4 Render nuclear/mesangial confounders as a distinct purple overlay in `feature_review.html`.
- [x] 9.5 Extend the operator adjudication template without overwriting existing user responses.
- [x] 9.6 Add tests for nuclear/mesangial false-slit separation.
- [x] 9.7 Rerun focused tests, full tests, OpenSpec validation, and the full quantification workflow.
- [x] 9.8 Record revised slit specificity, candidate metrics, and remaining limitations in `audit-results.md`.

## 10. Runtime Logging Improvement

- [x] 10.1 Add stage-level logs to manifest quantification showing manifest root, output root, `stop_after`, scored-example rows, ROI rows, embedding rows/columns, and final artifact count.
- [x] 10.2 Add burden-model logs showing grouped output directories, grouping key, subject/sample counts, fold sizes, prediction/metric/model paths, and candidate-screen paths.
- [x] 10.3 Add morphology feature extraction progress logs with row count, elapsed time, and rows-per-minute so long feature extraction stages are not silent.
- [x] 10.4 Add morphology review generation logs for case selection, asset-rendering progress, adjudication summary, and HTML output.
- [x] 10.5 Remove the pandas `FutureWarning` from adjudication yes-count summaries.
- [x] 10.6 Rerun changed-file lint and focused quantification tests.
- [x] 10.7 Rerun the full YAML workflow and record whether the logging is informative enough for runtime monitoring.

## 11. Border And False-Slit Readiness Blocker

- [x] 11.1 Inspect morphology review assets directly and record that accepted red slit calls are frequently glomerular-boundary/capsule artifacts or mesangial/nuclear structures.
- [x] 11.2 Quantify current review-case boundary contamination before revision.
- [x] 11.3 Add border false-slit feature columns and exclude boundary-adjacent slit candidates from accepted slit area/count features.
- [x] 11.4 Render border false-slit candidates separately in the review overlay.
- [x] 11.5 Extend the operator adjudication template with `border_false_slit_present` without overwriting existing user responses.
- [x] 11.6 Add feature-readiness gates to `morphology_candidate_summary.json` so morphology candidates are blocked when slit features fail visual/biological plausibility checks.
- [x] 11.7 Regenerate morphology feature and review artifacts from existing ROI crops.
- [x] 11.8 Rerun focused tests, OpenSpec validation, and record the revised verdict in `audit-results.md`.
