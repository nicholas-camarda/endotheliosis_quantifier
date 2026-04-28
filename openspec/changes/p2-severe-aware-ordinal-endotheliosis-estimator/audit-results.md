# P2 Audit Results

## Helper Reuse Decision

Inspected `src/eq/quantification/burden.py` functions `validate_score_values`, `score_to_stage_index`, `threshold_targets`, `burden_from_threshold_probabilities`, `monotonic_threshold_probabilities`, `score_probabilities_from_thresholds`, and `prediction_sets_from_score_probabilities`, plus `evaluate_burden_index_table` and `_write_threshold_support`. P2 reuses the low-level rubric and cumulative-threshold helpers, but keeps the evaluator contained in `src/eq/quantification/severe_aware_ordinal_estimator.py` because the existing burden evaluator is embedding-output specific and writes a different artifact schema.

Inspected `src/eq/quantification/source_aware_estimator.py` functions `source_aware_output_paths`, `_candidate_specs`, `_fit_predict_ridge`, `_raise_if_subject_leakage`, `_metric_row`, `_source_sensitivity`, and `evaluate_source_aware_endotheliosis_estimator`. P1 warning behavior remains a scope limiter, not a hard blocker, unless selected predictions are nonfinite or grouped validation leaks subjects.

## P1 Runtime Handoff

Selected P1 image candidate: `pooled_hybrid`.
Selected P1 subject candidate: `subject_source_adjusted_hybrid`.
Testing status: `testing_not_available_current_data_sensitivity`.
Scope limiters: `score_2_like_transitional_region, missing_optional_identity_column:image_id`.
High-score behavior: `{'score_2': {'row_count': 2, 'subject_count': 2, 'stage_index_mae': 0.9810719381170117, 'mean_predicted_stage_index': 79.55946860242354}, 'score_3': {'row_count': 1, 'subject_count': 1, 'stage_index_mae': 2.1621621621621614, 'mean_predicted_stage_index': 97.83783783783784}}`.

## Threshold Support Decision

`score >= 1.5`: `estimable` with 5 positive rows and 3 positive subjects.
`score >= 2`: `estimable_source_sensitive` with 3 positive rows and 2 positive subjects.
`score >= 3`: `non_estimable` with 1 positive rows and 1 positive subjects; it is not eligible as a reportable selected threshold because positives are confined to one source tail stratum.

## Feature Family Decision

ROI/QC and deterministic morphology are eligible for first-pass severe-aware candidate selection. Learned ROI and embedding-heavy inputs are audit comparators only because P1 showed strong training/apparent fit with degraded subject-heldout validation.

## Severe Failure Localization And Upstream Escalation

Current severe false negatives have ROI and mask provenance, so the first-pass failure localizes to current feature/model limitations rather than a proven missing-ROI failure. Targeted manual annotation and segmentation-backbone comparison are deferred unless the severe false-negative review identifies recurrent bad ROI extraction or mask geometry errors.

MedSAM/SAM audit is deferred to a separate upstream comparison. If opened, it must split oracle-box evidence from automatic-prompt evidence and measure segmentation metrics, prompt failures, ROI-feature stability, and downstream severe false-negative behavior.

Current feasibility inventory: FastAI U-Net, torchvision DeepLabV3, torch, opencv, and albumentations are installed in `eq-mac`; nnU-Net v2, DeepLabV3+ dependency stacks, Mask2Former-style stacks, SAM, and MedSAM are not installed and require separate environment/weights planning before they can be treated as runtime dependencies.

Alternative segmentation could matter only if changed masks alter severity-relevant ROI/morphology features such as lumen openness, area/fill fraction, component topology, pale/open-space features, slit-like structure, boundary fragmentation, or closed/open-lumen proxies and those changes reduce score >= 2 false negatives.

## Logging Participation

`src/eq/utils/execution_logging.py` is present. P2 participates as `high_level_function_events_only`: durable capture is owned by `endotheliosis_quantification` through `eq run-config --config configs/endotheliosis_quantification.yaml`; P2 adds no independent log root, file handlers, or tee implementation.
