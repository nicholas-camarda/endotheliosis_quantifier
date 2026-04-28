# P2 Audit Results

## Helper Reuse Decision

Inspected `src/eq/quantification/burden.py` functions `validate_score_values`, `score_to_stage_index`, `threshold_targets`, `burden_from_threshold_probabilities`, `monotonic_threshold_probabilities`, `score_probabilities_from_thresholds`, and `prediction_sets_from_score_probabilities`, plus `evaluate_burden_index_table` and `_write_threshold_support`. P2 reuses the low-level rubric and cumulative-threshold helpers, but keeps the evaluator contained in `src/eq/quantification/severe_aware_ordinal_estimator.py` because the existing burden evaluator is embedding-output specific and writes a different artifact schema.

Inspected `src/eq/quantification/source_aware_estimator.py` functions `source_aware_output_paths`, `_candidate_specs`, `_fit_predict_ridge`, `_raise_if_subject_leakage`, `_metric_row`, `_source_sensitivity`, and `evaluate_source_aware_endotheliosis_estimator`. P1 warning behavior remains a scope limiter, not a hard blocker, unless selected predictions are nonfinite or grouped validation leaks subjects.

## P1 Runtime Handoff

Selected P1 image candidate: `pooled_roi_qc`.
Selected P1 subject candidate: `subject_source_adjusted_hybrid`.
Testing status: `testing_not_available_current_data_sensitivity`.
Scope limiters: `wide_uncertainty, current_data_leave_source_out_sensitivity_only, numerical_warning_scope_limiter`.
High-score behavior: `score_2` has 81 rows / 36 subjects with mean predicted stage index 37.85 and stage-index MAE 42.15; `score_3` has 22 rows / 10 subjects with mean predicted stage index 37.32 and stage-index MAE 62.68.

## Threshold Support Decision

`score >= 1.5`: `estimable` with 196 positive rows and 50 positive subjects across `lauren_preeclampsia` and `vegfri_dox`.
`score >= 2`: `estimable_source_sensitive` with 103 positive rows and 37 positive subjects, all from `vegfri_dox`.
`score >= 3`: `exploratory` with 22 positive rows and 10 positive subjects, all from `vegfri_dox`; it is not eligible as a reportable primary threshold because positives are confined to one source tail stratum.

## Feature Family Decision

ROI/QC and deterministic morphology are eligible for first-pass severe-aware candidate selection. Learned ROI and embedding-heavy inputs are audit comparators only because P1 showed strong training/apparent fit with degraded subject-heldout validation.

## Severe Failure Localization And Upstream Escalation

Current severe false negatives have ROI and mask provenance, and manual review did not identify recurrent glomerulus/non-glomerulus mask failure as the primary issue. The failure therefore localizes to current feature/model limitations and grade-label ambiguity rather than a proven missing-ROI or mask-geometry failure. Segmentation-backbone comparison is not the immediate next fix unless a future audit shows that changed masks alter severity-relevant within-glomerulus features.

MedSAM/SAM audit is deferred to a separate upstream comparison. If opened, it must split oracle-box evidence from automatic-prompt evidence and measure segmentation metrics, prompt failures, ROI-feature stability, and downstream severe false-negative behavior.

Current feasibility inventory: FastAI U-Net, torchvision DeepLabV3, torch, opencv, and albumentations are installed in `eq-mac`; nnU-Net v2, DeepLabV3+ dependency stacks, Mask2Former-style stacks, SAM, and MedSAM are not installed and require separate environment/weights planning before they can be treated as runtime dependencies.

Alternative segmentation could matter only if changed masks alter severity-relevant ROI/morphology features such as lumen openness, area/fill fraction, component topology, pale/open-space features, slit-like structure, boundary fragmentation, or closed/open-lumen proxies and those changes reduce score >= 2 false negatives.

## Final Full-Cohort P2 Runtime Findings

Output root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/severe_aware_ordinal_estimator`.

Final severe-aware verdict: `limited_current_data_severe_aware_estimator`. Selected image candidate: `two_stage_severe_gate_roi_qc_morphology`. Selected subject candidate: `subject_severe_aware_ordinal`. Selected severe threshold: `score >= 2`. Reportable scopes are severe-risk label, subject-level summary, and aggregate current-data evidence; ordinal prediction sets and scalar burden are not reportable. README snippet eligibility remains `false`.

Selected-threshold performance is weak for the intended severe-aware purpose. At `score >= 2`, severe recall is 0.146, precision is 0.385, and 88/103 severe-positive images remain false negatives. At `score >= 1.5`, severe recall is 0.168 with 163/196 false negatives. At `score >= 3`, recall is 0.227 with 17/22 false negatives, but this tail threshold is exploratory because all positives come from `vegfri_dox`.

Reviewer adjudication evidence was persisted under `evidence/severe_false_negative_adjudications.json` and summarized under `evidence/severe_false_negative_adjudication_summary.json`. The corrected review contains 88 reviewed severe false negatives: 45 grade-correct, 32 grade-too-high, 10 grade-too-low, and 1 visually ambiguous. The corrected review has no inconsistent `grade_too_high` plus `valid_grade_model_miss` rows. After adjudication, 56 reviewed false negatives remain adjudicated severe and 32 are removed from the severe-positive set. Recomputed selected-threshold metrics under the adjudicated false-negative labels are recall 0.211, precision 0.385, and 56/71 severe-positive images still false negative.

The adjudicated-label rerun is contained under `adjudicated_label_rerun/` and is severe-threshold scoped only. Its selected candidate is `adjudicated_severe_roi_qc_morphology_threshold`; validation recall is 0.113, precision is 0.400, and 63/71 adjudicated severe-positive images remain false negatives. This confirms that adjudication cleans up some label noise but does not make the current ROI/QC/morphology features adequate for grade 2/3 detection.

Final manifest status is complete with no missing or unmanifested artifacts. The runtime rerun emitted nonfatal scikit-learn numerical warnings, which remain scope limiters rather than hard blockers because selected output artifacts are finite.

## Logging Participation

`src/eq/utils/execution_logging.py` is present. P2 participates as `high_level_function_events_only`: durable capture is owned by `endotheliosis_quantification` through `eq run-config --config configs/endotheliosis_quantification.yaml`; P2 adds no independent log root, file handlers, or tee implementation.
