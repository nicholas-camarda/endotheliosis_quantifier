## 1. Provider Audit And Output Contract

- [ ] 1.1 Create `src/eq/quantification/learned_roi.py` with constants for `burden_model/learned_roi/` output groups: `primary_model`, `validation`, `calibration`, `summaries`, `evidence`, `candidates`, `diagnostics`, and `feature_sets`.
- [ ] 1.2 Implement provider audit for `current_glomeruli_encoder`, `simple_roi_qc`, `torchvision_resnet18_imagenet`, and `timm_dino_or_vit`.
- [ ] 1.3 Write `burden_model/learned_roi/diagnostics/provider_audit.json` with provider availability, package provenance, model/weight provenance, failure messages, and network/download requirements.
- [ ] 1.4 Ensure optional providers marked unavailable are not silently substituted and do not block baseline providers.
- [ ] 1.5 Add unit tests for provider audit schema and unavailable-provider behavior.

## 2. Learned ROI Feature Extraction

- [ ] 2.1 Implement learned feature extraction from existing ROI crop rows and current embedding table.
- [ ] 2.2 Reuse existing `embeddings/roi_embeddings.csv` as the `current_glomeruli_encoder` feature provider without recomputing segmentation embeddings unnecessarily.
- [ ] 2.3 Implement `simple_roi_qc` features as low-dimensional non-mechanistic QC/coarse features using ROI image, ROI mask, stain/intensity, mask area/shape, and quality signals.
- [ ] 2.4 Implement optional `torchvision_resnet18_imagenet` feature extraction only when local imports and weights are available without downloads.
- [ ] 2.5 Implement optional `timm_dino_or_vit` feature extraction only when local imports and weights are available without downloads.
- [ ] 2.6 Write `burden_model/learned_roi/feature_sets/learned_roi_features.csv`.
- [ ] 2.7 Write `burden_model/learned_roi/feature_sets/learned_roi_feature_metadata.json`.
- [ ] 2.8 Write `burden_model/learned_roi/diagnostics/learned_roi_feature_diagnostics.json`.
- [ ] 2.9 Add unit tests for learned feature schema, identity preservation, finite feature outputs, provider prefixes, and diagnostics.

## 3. Learned ROI Candidate Modeling

- [ ] 3.1 Add image-level learned ROI candidate screens with subject-heldout folds.
- [ ] 3.2 Add subject-level learned ROI candidate screens using one aggregated row per `subject_id`.
- [ ] 3.3 Evaluate candidate families: provider-only, simple-QC-only, provider-plus-simple-QC, and provider-plus-morphology-QC only when morphology readiness status is carried forward.
- [ ] 3.4 Write `burden_model/learned_roi/candidates/learned_roi_candidate_metrics.csv`.
- [ ] 3.5 Write `burden_model/learned_roi/validation/learned_roi_predictions.csv`.
- [ ] 3.6 Write `burden_model/learned_roi/candidates/learned_roi_candidate_summary.json` with per-image readiness, subject/cohort readiness, blockers, and next action.
- [ ] 3.7 Ensure candidate selection never promotes a model solely because it has the lowest MAE.
- [ ] 3.8 Add tests proving image candidates are subject-heldout and subject candidates use one row per `subject_id`.

## 4. Calibration And Uncertainty

- [ ] 4.1 Add grouped conformal prediction-set calibration for image-level learned ROI candidates.
- [ ] 4.2 Add continuous burden/stage-index interval calibration where applicable.
- [ ] 4.3 Write `burden_model/learned_roi/calibration/learned_roi_calibration.json`.
- [ ] 4.4 Report prediction-set coverage and average prediction-set size overall, by cohort, and by observed score where estimable.
- [ ] 4.5 Compare learned ROI prediction-set size against the current broad baseline and record whether uncertainty materially narrowed without undercoverage.
- [ ] 4.6 Add tests for calibration artifact schema, score-set validity, and grouped calibration provenance.

## 5. Learned ROI Review Evidence

- [ ] 5.1 Create `src/eq/quantification/learned_roi_review.py`.
- [ ] 5.2 Write `burden_model/learned_roi/evidence/learned_roi_review.html`.
- [ ] 5.3 Write `burden_model/learned_roi/evidence/learned_roi_review_examples.csv`.
- [ ] 5.4 Write `burden_model/learned_roi/evidence/learned_roi_nearest_examples.csv` with same-subject nearest neighbors excluded for held-out predictions.
- [ ] 5.5 Render review assets under `burden_model/learned_roi/evidence/assets/`.
- [ ] 5.6 If saliency or attention artifacts are available for a provider, label them as heuristic model-support visualizations; otherwise record attribution as unavailable.
- [ ] 5.7 Add tests proving review HTML, examples, nearest-neighbor evidence, and asset links are written.

## 6. Workflow And Report Integration

- [ ] 6.1 Integrate learned ROI evaluation into `src/eq/quantification/pipeline.py` after ROI embeddings and before combined review generation.
- [ ] 6.2 Keep `eq run-config --config configs/endotheliosis_quantification.yaml` as the reproducible front door.
- [ ] 6.3 Keep `eq quant-endo` as the direct one-shot CLI; do not add a new user-facing quantification command.
- [ ] 6.4 Update `quantification_review/quantification_review.html` to include learned ROI provider audit, candidate summary, readiness, evidence links, and claim boundary.
- [ ] 6.5 Update `quantification_review/results_summary.md`, `results_summary.csv`, and `readme_results_snippet.md` so learned ROI results appear only if a track passes readiness gates.
- [ ] 6.6 Ensure deterministic morphology features are presented as blocked QC/evidence when `morphology_candidate_summary.json` reports `blocked_by_visual_feature_readiness`.
- [ ] 6.7 Add integration tests proving the YAML quantification workflow writes learned ROI artifacts under grouped paths and no flat learned ROI aliases.

## 7. Documentation And Claim Boundary

- [ ] 7.1 Update `README.md` only with current-state learned ROI wording if a readiness gate passes; otherwise state that quantification is exploratory.
- [ ] 7.2 Update `docs/OUTPUT_STRUCTURE.md` with the `burden_model/learned_roi/` grouped artifact layout.
- [ ] 7.3 Update `docs/TECHNICAL_LAB_NOTEBOOK.md` with the learned ROI target, validation design, and morphology-pivot rationale.
- [ ] 7.4 Update `docs/ONBOARDING_GUIDE.md` only if the command surface or artifact review workflow changes.
- [ ] 7.5 Ensure docs do not describe learned ROI outputs as true tissue percent, closed-capillary percent, causal evidence, or mechanistic proof.

## 8. Runtime Review And Final Validation

- [ ] 8.1 Run changed-file lint and formatting checks.
- [ ] 8.2 Run focused learned ROI, burden, pipeline, and review tests.
- [ ] 8.3 Run the full test suite with `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 8.4 Run `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [ ] 8.5 Inspect provider audit, learned feature diagnostics, learned candidate metrics, calibration, evidence HTML, nearest examples, combined quantification review, and README snippet.
- [ ] 8.6 Record exact artifact links, provider availability, candidate results, readiness verdict, what worked, what failed, and next action in `audit-results.md`.
- [ ] 8.7 Run `openspec validate p0-learned-roi-endotheliosis-quantification --strict`.
- [ ] 8.8 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p0-learned-roi-endotheliosis-quantification`.
