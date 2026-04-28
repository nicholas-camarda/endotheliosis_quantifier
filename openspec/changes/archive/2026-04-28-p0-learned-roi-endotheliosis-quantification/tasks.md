## 1. Provider Audit And Output Contract

- [x] 1.1 Create `src/eq/quantification/learned_roi.py` with constants for `burden_model/learned_roi/` output groups: `primary_model`, `validation`, `calibration`, `summaries`, `evidence`, `candidates`, `diagnostics`, and `feature_sets`.
- [x] 1.2 Implement provider audit for `current_glomeruli_encoder`, `simple_roi_qc`, `torchvision_resnet18_imagenet`, and `timm_dino_or_vit`.
- [x] 1.3 Write `burden_model/learned_roi/diagnostics/provider_audit.json` with provider availability, package provenance, model/weight provenance, failure messages, and network/download requirements.
- [x] 1.4 Use provider statuses `available_fit_allowed`, `available_audit_only`, `unavailable`, and `failed`; only `current_glomeruli_encoder` and `simple_roi_qc` may be `available_fit_allowed` in phase 1.
- [x] 1.5 Ensure optional providers marked unavailable or audit-only are not silently substituted, are not fitted in phase 1, and do not block baseline providers.
- [x] 1.6 Add unit tests for provider audit schema, audit-only provider behavior, unavailable-provider behavior, and phase-1 fit eligibility.

## 2. Learned ROI Feature Extraction

- [x] 2.1 Implement learned feature extraction from existing ROI crop rows and current embedding table.
- [x] 2.2 Reuse existing `embeddings/roi_embeddings.csv` as the `current_glomeruli_encoder` feature provider without recomputing segmentation embeddings unnecessarily.
- [x] 2.3 Implement `simple_roi_qc` features as low-dimensional non-mechanistic QC/coarse features using ROI image, ROI mask, stain/intensity, mask area/shape, and quality signals.
- [x] 2.4 Implement optional `torchvision_resnet18_imagenet` feature extraction only when local imports and weights are available without downloads.
- [x] 2.5 Implement optional `timm_dino_or_vit` feature extraction only when local imports and weights are available without downloads.
- [x] 2.6 Write `burden_model/learned_roi/feature_sets/learned_roi_features.csv`.
- [x] 2.7 Write `burden_model/learned_roi/feature_sets/learned_roi_feature_metadata.json`.
- [x] 2.8 Write `burden_model/learned_roi/diagnostics/learned_roi_feature_diagnostics.json`.
- [x] 2.9 Add unit tests for learned feature schema, identity preservation, finite feature outputs, provider prefixes, and diagnostics.

## 3. Learned ROI Candidate Modeling

- [x] 3.1 Add image-level learned ROI candidate screens with subject-heldout folds.
- [x] 3.2 Add subject-level learned ROI candidate screens using one aggregated row per `subject_id`.
- [x] 3.3 Evaluate only the fixed phase-1 candidate IDs: `image_current_glomeruli_encoder`, `image_simple_roi_qc`, `image_current_glomeruli_encoder_plus_simple_roi_qc`, `subject_current_glomeruli_encoder`, `subject_simple_roi_qc`, and `subject_current_glomeruli_encoder_plus_simple_roi_qc`.
- [x] 3.4 Write `burden_model/learned_roi/candidates/learned_roi_candidate_metrics.csv`.
- [x] 3.5 Write `burden_model/learned_roi/validation/learned_roi_predictions.csv`.
- [x] 3.6 Write `burden_model/learned_roi/candidates/learned_roi_candidate_summary.json` with per-image readiness, subject/cohort readiness, blockers, and next action.
- [x] 3.7 Ensure candidate selection never promotes a model solely because it has the lowest MAE or solely because it improves the 0-100 stage-index recoding.
- [x] 3.8 Report ordinal/grade-scale metrics side by side with stage-index metrics for every fitted candidate.
- [x] 3.9 Add tests proving image candidates are subject-heldout, subject candidates use one row per `subject_id`, and optional audit-only providers produce no fitted candidate rows.

## 3A. Cohort Confounding Diagnostics

- [x] 3A.1 Write `burden_model/learned_roi/diagnostics/cohort_confounding_diagnostics.json`.
- [x] 3A.2 Include row counts, `subject_id` counts, and score distributions by `cohort_id`.
- [x] 3A.3 Include candidate residual summaries, prediction-set coverage, and average prediction-set size by cohort.
- [x] 3A.4 Add a selected-feature cohort-predictability screen with cross-validated balanced accuracy.
- [x] 3A.5 Add leave-one-cohort-out or train-one-cohort/test-other diagnostics where score support is sufficient.
- [x] 3A.6 Block README/docs readiness when cohort-specific coverage is `<0.80` for a cohort with at least 30 rows, cohort grade-scale MAE differs by `>=0.35`, cohort balanced accuracy is `>=0.80` while score prediction remains weak or unstable, or leave-one-cohort-out diagnostics collapse.
- [x] 3A.7 Add tests for cohort diagnostic schema and readiness-blocker propagation.

## 4. Calibration And Uncertainty

- [x] 4.1 Add grouped conformal prediction-set calibration for image-level learned ROI candidates.
- [x] 4.2 Add continuous burden/stage-index interval calibration where applicable.
- [x] 4.3 Write `burden_model/learned_roi/calibration/learned_roi_calibration.json`.
- [x] 4.4 Report prediction-set coverage and average prediction-set size overall, by cohort, and by observed score where estimable.
- [x] 4.5 Compare learned ROI prediction-set size against the current baseline average set size `5.308` and record whether the selected image candidate reaches average set size `<=4.0`.
- [x] 4.6 Enforce image-track readiness thresholds: overall empirical coverage `>=0.88`, observed-score coverage `>=0.80` for strata with at least 30 rows, cohort coverage `>=0.80` for cohorts with at least 30 rows, average set size `<=4.0`, no nonfinite selected features or predictions, no unresolved numerical-instability warnings, and no cohort-confounding blocker.
- [x] 4.7 Enforce subject/cohort-track readiness thresholds: one row per `subject_id`, subject-heldout validation, grouped bootstrap intervals, no nonfinite predictions, no unresolved numerical-instability warnings, no cohort-confounding blocker, and explicit no-per-image-precision claim text.
- [x] 4.8 Add tests for calibration artifact schema, score-set validity, grouped calibration provenance, and readiness-threshold blocking.

## 5. Learned ROI Review Evidence

- [x] 5.1 Create `src/eq/quantification/learned_roi_review.py`.
- [x] 5.2 Write `burden_model/learned_roi/evidence/learned_roi_review.html`.
- [x] 5.3 Write `burden_model/learned_roi/evidence/learned_roi_review_examples.csv`.
- [x] 5.4 Write `burden_model/learned_roi/evidence/learned_roi_nearest_examples.csv` with same-subject nearest neighbors excluded for held-out predictions.
- [x] 5.5 Render review assets under `burden_model/learned_roi/evidence/assets/`.
- [x] 5.6 If saliency or attention artifacts are available for a provider, label them as heuristic model-support visualizations; otherwise record attribution as unavailable.
- [x] 5.7 Add tests proving review HTML, examples, nearest-neighbor evidence, and asset links are written.

## 6. Workflow And Report Integration

- [x] 6.1 Integrate learned ROI evaluation into `src/eq/quantification/pipeline.py` after ROI embeddings and before combined review generation.
- [x] 6.2 Keep `eq run-config --config configs/endotheliosis_quantification.yaml` as the reproducible front door.
- [x] 6.3 Keep `eq quant-endo` as the direct one-shot CLI; do not add a new user-facing quantification command.
- [x] 6.4 Update `quantification_review/quantification_review.html` to include learned ROI provider audit, candidate summary, readiness, evidence links, and claim boundary.
- [x] 6.5 Update `quantification_review/results_summary.md`, `results_summary.csv`, and `readme_results_snippet.md` so learned ROI results appear only if a track passes readiness gates.
- [x] 6.6 Include cohort-confounding diagnostic status in the combined review and block README snippet learned ROI promotion when cohort diagnostics fail.
- [x] 6.7 Ensure deterministic morphology features are presented as blocked QC/evidence when `morphology_candidate_summary.json` reports `blocked_by_visual_feature_readiness`.
- [x] 6.8 Add integration tests proving the YAML quantification workflow writes learned ROI artifacts under grouped paths and no flat learned ROI aliases.

## 7. Documentation And Claim Boundary

- [x] 7.1 Update `README.md` only with current-state learned ROI wording if a readiness gate passes; otherwise state that quantification is exploratory.
- [x] 7.2 Update `docs/OUTPUT_STRUCTURE.md` with the `burden_model/learned_roi/` grouped artifact layout.
- [x] 7.3 Update `docs/TECHNICAL_LAB_NOTEBOOK.md` with the learned ROI target, validation design, and morphology-pivot rationale.
- [x] 7.4 Update `docs/ONBOARDING_GUIDE.md` only if the command surface or artifact review workflow changes.
- [x] 7.5 Ensure docs do not describe learned ROI outputs as true tissue percent, closed-capillary percent, causal evidence, or mechanistic proof.

## 8. Runtime Review And Final Validation

- [x] 8.1 Run changed-file lint and formatting checks.
- [x] 8.2 Run focused learned ROI, burden, pipeline, and review tests.
- [x] 8.3 Run the full test suite with `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 8.4 Run `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 8.5 Inspect provider audit, learned feature diagnostics, learned candidate metrics, calibration, evidence HTML, nearest examples, combined quantification review, and README snippet.
- [x] 8.6 Record exact artifact links, provider availability, candidate results, readiness verdict, what worked, what failed, and next action in `audit-results.md`.
- [x] 8.7 If readiness gates remain unmet, verify the failure evidence is complete and no README/docs-ready learned ROI claim is emitted.
- [x] 8.8 Run `openspec validate p0-learned-roi-endotheliosis-quantification --strict`.
- [x] 8.9 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p0-learned-roi-endotheliosis-quantification`.
