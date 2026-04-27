## Why

The current burden-index workflow has the right score, identity, validation, and reporting contracts, but its feature representation is not yet aligned with how endotheliosis is graded. Endotheliosis severity is assessed from the relative amount of open versus collapsed capillary/arteriole lumina in a glomerulus, and current frozen embeddings plus coarse ROI scalar features do not explicitly model open lumina, collapsed slit-like structures, or erythrocyte-filled patent lumina.

This change creates the next quantification feature layer: morphology-aware features, feature QA, artifact organization, and subject/cohort burden modeling that can be reviewed before any README/docs-ready claim.

## What Changes

- Reorganize the `burden_model/` output contract into meaningful subfolder groups so primary model outputs, validation, calibration, summaries, evidence, candidates, diagnostics, and feature sets cannot be confused.
- Add a morphology-aware feature extraction module under `src/eq/quantification/` that computes explicit open-lumen, collapsed/slit/ridge, erythrocyte-confounder, and image-quality/orientation features from the existing union ROI image/mask crops.
- Add feature QA artifacts that visualize raw ROI, glomerulus mask, detected pale/open lumina, detected erythrocyte-filled patent lumen candidates, detected collapsed/slit-like structures, feature values, and manual score.
- Add an operator adjudication loop for difficult morphology examples so the user can easily help label a small review set as plug-and-play evidence rather than editing code or CSVs manually.
- Promote subject-level morphology aggregation into a first-class cohort-summary candidate with subject-heldout validation, grouped bootstrap confidence intervals, and track-specific readiness gates.
- Add per-image morphology-aware candidate models to test whether explicit morphology features narrow prediction sets while preserving nominal coverage.
- Add feature and numerical diagnostics for morphology features, frozen embeddings, and combined feature matrices before model fitting.
- Update `quantification_review/` outputs so users can inspect which biological features drove subject/cohort summaries and where RBC-filled patent lumina or poor slice orientation complicate interpretation.
- Keep the current burden-index model and ordinal outputs as comparators; do not replace the current burden-index contract without refreshed evidence.
- **BREAKING**: current flat `burden_model/*` runtime output paths will be superseded by grouped subfolder paths for new runs. Existing historical runtime outputs remain historical artifacts and should not be compatibility-shimmed into the new layout.

## Capabilities

### New Capabilities

- `morphology-aware-quantification-features`: Defines open-lumen, collapsed/slit/ridge, erythrocyte-confounder, and quality/orientation feature extraction; feature QA panels; operator adjudication artifacts; and morphology-aware image/subject candidate screens.

### Modified Capabilities

- `endotheliosis-burden-index`: Updates the burden output layout, candidate-screen contract, subject/cohort summary readiness gates, and review report requirements to incorporate morphology-aware features and organized artifact groups.

## Impact

- Affected modules:
  - `src/eq/quantification/pipeline.py`
  - `src/eq/quantification/burden.py`
  - `src/eq/quantification/morphology_features.py`
  - `src/eq/quantification/feature_review.py`
- Affected command:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`
- Affected output root:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`
- New grouped `burden_model/` layout for future runs:
  - `burden_model/primary_model/`
  - `burden_model/validation/`
  - `burden_model/calibration/`
  - `burden_model/summaries/`
  - `burden_model/evidence/`
  - `burden_model/candidates/`
  - `burden_model/diagnostics/`
  - `burden_model/feature_sets/`
- New feature artifacts:
  - `burden_model/feature_sets/morphology_features.csv`
  - `burden_model/feature_sets/morphology_feature_metadata.json`
  - `burden_model/feature_sets/subject_morphology_features.csv`
  - `burden_model/diagnostics/morphology_feature_diagnostics.json`
  - `burden_model/evidence/morphology_feature_review/feature_review.html`
  - `burden_model/evidence/morphology_feature_review/feature_review_cases.csv`
  - `burden_model/evidence/morphology_feature_review/assets/`
  - `burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv`
- New or updated candidate artifacts:
  - `burden_model/candidates/morphology_candidate_metrics.csv`
  - `burden_model/candidates/subject_morphology_candidate_predictions.csv`
  - `burden_model/candidates/morphology_candidate_summary.json`
- Tests:
  - unit tests for morphology feature schema, bounds, finite outputs, and RBC-confounder feature fields
  - fixture tests for feature-review artifact generation
  - focused pipeline tests for grouped burden output layout
  - regression tests that candidate artifacts are labeled as candidates and not deployed models
- Scientific claim boundary:
  - Morphology features are predictive support features for endotheliosis burden, not causal proof of a biological mechanism.
  - RBC-filled patent lumina are a named confounder: dark or filled lumina must not be treated as collapsed solely because they are not pale/empty.
  - Subject/cohort burden summaries remain distinct from individual image prediction readiness.

## Explicit Decisions

- Change name: `p0-build-morphology-aware-quantification-features`.
- New feature module: `src/eq/quantification/morphology_features.py`.
- New feature-review module: `src/eq/quantification/feature_review.py`.
- Existing workflow config remains `configs/endotheliosis_quantification.yaml`.
- Existing CLI surfaces remain `eq run-config --config configs/endotheliosis_quantification.yaml`, `eq quant-endo`, and `eq prepare-quant-contract`.
- Current primary image-level comparator remains `src/eq/quantification/burden.py` until refreshed morphology-aware evidence selects a better operational path.
- Output organization is part of this change. New runs SHALL write grouped subfolders under `burden_model/`; the implementation SHALL NOT add compatibility shims that pretend old flat outputs are current.
- The first operator-assisted review artifact SHALL be a CSV template at `burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv` plus an HTML review page. The user should be able to fill simple columns, save the CSV, and rerun the same YAML.
- Operator review labels SHALL include at least `open_empty_lumen_present`, `open_rbc_filled_lumen_present`, `collapsed_slit_like_lumen_present`, `poor_orientation_or_quality`, `feature_detection_problem`, and `notes`.
- RBC-filled patent lumina SHALL be represented as a confounder class in feature QA and candidate interpretation.
- Subject/cohort morphology summaries SHALL validate by `subject_id`; image-level morphology candidates SHALL use subject-heldout folds.

## Open Questions

- [audit_first_then_decide] The exact image-processing thresholds for pale lumen detection, erythrocyte color detection, and collapsed-line detection SHALL be selected from feature QA evidence generated on the current scored cohort rather than guessed from theory.
- [audit_first_then_decide] Whether morphology features alone, morphology plus frozen embeddings, or subject-level aggregation should be selected for README/docs-ready summaries SHALL be decided from the refreshed candidate artifacts.
- [defer_ok] Learned morphology encoders, self-supervised ROI models, and mitochondria-transfer feature comparisons may be added after explicit morphology features and QA panels establish the baseline feature contract.
